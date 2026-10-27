#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, const char **argv) {
    @autoreleasepool {
        if (argc != 4) return 2;
        const unsigned size = (unsigned)strtoul(argv[3], NULL, 10);
        if (size > 6) return 2;
        NSError *error = nil;
        NSString *source = [NSString stringWithContentsOfFile:@(argv[1])
                                                    encoding:NSUTF8StringEncoding error:&error];
        NSData *data = [NSData dataWithContentsOfFile:@(argv[2]) options:0 error:&error];
        NSDictionary *metadata = data
            ? [NSJSONSerialization JSONObjectWithData:data options:0 error:&error] : nil;
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!source || !metadata || !device) {
            fprintf(stderr, "storage-image inputs/device unavailable: %s\n",
                    error.description.UTF8String);
            return 1;
        }
        MTLCompileOptions *options = [MTLCompileOptions new];
        if (@available(macOS 15.0, *)) {
            options.mathMode = MTLMathModeSafe;
        } else {
            fprintf(stderr, "this safe-math storage-image probe requires macOS 15 or newer\n");
            return 1;
        }
        id<MTLLibrary> library = [device newLibraryWithSource:source options:options error:&error];
        id<MTLFunction> function = [library newFunctionWithName:metadata[@"entry"]];
        id<MTLComputePipelineState> pipeline = function
            ? [device newComputePipelineStateWithFunction:function error:&error] : nil;
        if (!pipeline) {
            fprintf(stderr, "storage-image compilation/pipeline: %s\n",
                    error.description.UTF8String);
            return 1;
        }
        const MTLPixelFormat formats[] = {
            MTLPixelFormatR8Uint, MTLPixelFormatR8Sint, MTLPixelFormatR16Uint,
            MTLPixelFormatR16Sint, MTLPixelFormatR32Uint,
            MTLPixelFormatRG32Uint, MTLPixelFormatRGBA32Uint,
        };
        const unsigned bytes = size < 2 ? 1 : size < 4 ? 2 : 4;
        const unsigned components = size == 5 ? 2 : size == 6 ? 4 : 1;
        const BOOL signedValue = size == 1 || size == 3;
        const uint32_t boundaries[] = {0x80000000, 0xffffffff, 0, 0x7fffffff};
        const uint32_t extra[] = {0x7fc00001, 0xff800000, 0x3f800000, 0x01234567};
        uint8_t initial[64] = {};
        uint32_t expected[4][4] = {};
        for (unsigned lane = 0; lane < 4; ++lane) {
            for (unsigned component = 0; component < components; ++component) {
                uint32_t value = component ? extra[(lane + component) % 4] : boundaries[lane];
                if (bytes < 4) value >>= 32 - bytes * 8;
                memcpy(initial + (lane * components + component) * bytes, &value, bytes);
                if (signedValue) {
                    value = bytes == 1 ? (uint32_t)(int32_t)(int8_t)value
                                       : (uint32_t)(int32_t)(int16_t)value;
                }
                expected[lane][component] = value;
            }
        }
        id<MTLArgumentEncoder> arguments = [function newArgumentEncoderWithBufferIndex:0];
        id<MTLBuffer> argumentBuffer = [device newBufferWithLength:arguments.encodedLength
                                                        options:MTLResourceStorageModeShared];
        [arguments setArgumentBuffer:argumentBuffer offset:0];
        NSArray *bindings = metadata[@"bindings"];
        if (bindings.count != 2 || !arguments || !argumentBuffer) {
            fprintf(stderr, "storage-image argument contract mismatch\n");
            return 1;
        }
        id<MTLTexture> textures[2] = {};
        id<MTLBuffer> backing[2] = {};
        for (unsigned index = 0; index < 2; ++index) {
            NSDictionary *binding = bindings[index];
            NSString *type = binding[@"msl_type"];
            MTLTextureDescriptor *descriptor = [MTLTextureDescriptor new];
            descriptor.textureType = [type containsString:@"texture_buffer"]
                ? MTLTextureTypeTextureBuffer : [type containsString:@"texture1d"]
                ? MTLTextureType1D : MTLTextureType2D;
            descriptor.width = 4;
            descriptor.height = 1;
            descriptor.pixelFormat = index ? MTLPixelFormatRGBA32Uint : formats[size];
            descriptor.storageMode = MTLStorageModeShared;
            descriptor.usage = index ? MTLTextureUsageShaderWrite : MTLTextureUsageShaderRead;
            if (descriptor.textureType == MTLTextureTypeTextureBuffer) {
                const NSUInteger alignment = MAX(256,
                    [device minimumTextureBufferAlignmentForPixelFormat:descriptor.pixelFormat]);
                backing[index] = [device newBufferWithLength:alignment
                                                    options:MTLResourceStorageModeShared];
                textures[index] = [backing[index] newTextureWithDescriptor:descriptor
                                                                  offset:0 bytesPerRow:alignment];
            } else {
                textures[index] = [device newTextureWithDescriptor:descriptor];
            }
            if (!textures[index]) {
                fprintf(stderr, "storage-image texture allocation failed\n");
                return 1;
            }
            if (!index) {
                if (backing[index]) memcpy(backing[index].contents, initial, sizeof initial);
                else [textures[index] replaceRegion:MTLRegionMake2D(0, 0, 4, 1) mipmapLevel:0
                    withBytes:initial bytesPerRow:4 * components * bytes];
            }
            [arguments setTexture:textures[index] atIndex:[binding[@"arg_id"] unsignedIntegerValue]];
        }
        id<MTLCommandQueue> queue = [device newCommandQueue];
        id<MTLCommandBuffer> command = [queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:argumentBuffer offset:0 atIndex:0];
        [encoder useResource:textures[0] usage:MTLResourceUsageRead];
        [encoder useResource:textures[1] usage:MTLResourceUsageWrite];
        [encoder dispatchThreadgroups:MTLSizeMake(1, 1, 1)
               threadsPerThreadgroup:MTLSizeMake(4, 1, 1)];
        [encoder endEncoding];
        [command commit];
        [command waitUntilCompleted];
        if (command.status != MTLCommandBufferStatusCompleted) {
            fprintf(stderr, "storage-image GPU execution: %s\n",
                    command.error.description.UTF8String);
            return 1;
        }
        uint32_t actual[4][4];
        if (backing[1]) memcpy(actual, backing[1].contents, sizeof actual);
        else [textures[1] getBytes:actual bytesPerRow:sizeof actual
                       fromRegion:MTLRegionMake2D(0, 0, 4, 1) mipmapLevel:0];
        if (memcmp(actual, expected, sizeof actual)) {
            for (unsigned lane = 0; lane < 4; ++lane) {
                for (unsigned component = 0; component < 4; ++component) {
                    if (actual[lane][component] != expected[lane][component])
                        fprintf(stderr, "lane %u component %u: %08x != %08x\n", lane, component,
                                actual[lane][component], expected[lane][component]);
                }
            }
            return 1;
        }
    }
    return 0;
}
