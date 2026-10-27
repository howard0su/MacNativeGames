#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import "nvn_refl.h"
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

static id<MTLBuffer> FixtureBuffer(id<MTLDevice> device, const nvn_refl *parsed, nvn_rbind binding) {
    id<MTLBuffer> buffer = [device newBufferWithLength:65536 options:MTLResourceStorageModeShared];
    if (!buffer) {
        fprintf(stderr, "cannot allocate GPU fixture buffer\n");
        return nil;
    }
    memset(buffer.contents, 0, buffer.length);
    if (binding.kind == NVN_RB_C1) {
        memcpy(buffer.contents, parsed->c1, parsed->c1_len);
    } else if (binding.kind == NVN_RB_CBUF && binding.reg == 0) {
        const uint32_t index = 7;
        memcpy(buffer.contents, &index, sizeof index);
        const uint32_t offsets[] = {0x00ff0001, 0xff000100};
        memcpy((char *)buffer.contents + 16, offsets, sizeof offsets);
        const uint32_t rounding[] = {0x3f800001, 0x3f7ffffe, 0xbf800000};
        memcpy((char *)buffer.contents + 32, rounding, sizeof rounding);
    } else if (binding.kind == NVN_RB_CBUF && binding.reg == 3) {
        const uint64_t guestAddress = UINT64_C(0x1234567000);
        memcpy(buffer.contents, &guestAddress, sizeof guestAddress);
    } else if (binding.kind == NVN_RB_SBUFU && binding.reg == 3 && binding.off == 0) {
        const uint32_t values[] = {0xdeadbeef, 0, 0x3f000000};
        memcpy(buffer.contents, values, sizeof values);
    } else if (binding.kind == NVN_RB_DRV0) {
        const uint64_t guestAddress = UINT64_C(0x9876543000);
        memcpy((char *)buffer.contents + 0x310, &guestAddress, sizeof guestAddress);
    } else if (binding.kind != NVN_RB_SBUFP || binding.reg != 0) {
        fprintf(stderr, "unexpected GPU fixture binding\n");
        return nil;
    }
    return buffer;
}

static int ComputeProbe(id<MTLDevice> device, id<MTLLibrary> library,
                        const nvn_refl *parsed, const char *expectedText) {
    NSError *error = nil;
    id<MTLFunction> function = [library newFunctionWithName:library.functionNames[0]];
    id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:function
                                                                               error:&error];
    if (!pipeline) {
        fprintf(stderr, "create compute pipeline: %s\n", error.description.UTF8String);
        return 1;
    }
    id<MTLArgumentEncoder> arguments = [function newArgumentEncoderWithBufferIndex:parsed->abi];
    id<MTLBuffer> table = [device newBufferWithLength:arguments.encodedLength
                                           options:MTLResourceStorageModeShared];
    id<MTLCommandQueue> queue = [device newCommandQueue];
    id<MTLCommandBuffer> commands = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [commands computeCommandEncoder];
    if (!arguments || !table || !encoder) {
        fprintf(stderr, "cannot allocate compute fixture resources\n");
        return 1;
    }
    [arguments setArgumentBuffer:table offset:0];
    NSMutableArray<id<MTLBuffer>> *resources = [NSMutableArray arrayWithObject:table];
    id<MTLBuffer> output = nil;
    [encoder setComputePipelineState:pipeline];
    for (int i = 0; i < parsed->nb; i++) {
        nvn_rbind binding = parsed->b[i];
        id<MTLBuffer> buffer = FixtureBuffer(device, parsed, binding);
        if (!buffer) return 1;
        if (binding.kind == NVN_RB_SBUFP && binding.reg == 0) output = buffer;
        [resources addObject:buffer];
        [arguments setBuffer:buffer offset:0 atIndex:binding.aid];
        [encoder useResource:buffer usage:MTLResourceUsageRead |
                                         (binding.rw ? MTLResourceUsageWrite : 0)];
    }
    if (!output || !parsed->tg[0] || !parsed->tg[1] || !parsed->tg[2]) {
        fprintf(stderr, "compute fixture needs an output buffer and workgroup size\n");
        return 1;
    }
    [encoder setBuffer:table offset:0 atIndex:parsed->abi];
    [encoder dispatchThreadgroups:MTLSizeMake(1, 1, 1)
           threadsPerThreadgroup:MTLSizeMake(parsed->tg[0], parsed->tg[1], parsed->tg[2])];
    [encoder endEncoding];
    [commands commit];
    [commands waitUntilCompleted];
    if (commands.status != MTLCommandBufferStatusCompleted) {
        fprintf(stderr, "compute GPU probe failed: %s\n", commands.error.description.UTF8String);
        return 1;
    }
    uint32_t value;
    memcpy(&value, output.contents, sizeof value);
    char *end = NULL;
    unsigned long expected = strtoul(expectedText, &end, 16);
    if (!expectedText[0] || *end || expected > UINT32_MAX || value != expected) {
        fprintf(stderr, "compute GPU result: %08x, expected %s\n", value, expectedText);
        return 1;
    }
    return 0;
}

int main(int argc, const char **argv) {
    @autoreleasepool {
        if (argc < 2 || argc > 5) return 2;
        NSError *error = nil;
        NSString *source = [NSString stringWithContentsOfFile:@(argv[1])
                                                   encoding:NSUTF8StringEncoding error:&error];
        if (!source) {
            fprintf(stderr, "read MSL: %s\n", error.description.UTF8String);
            return 1;
        }
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) {
            fprintf(stderr, "no Metal device available\n");
            return 1;
        }
        MTLCompileOptions *options = [MTLCompileOptions new];
        if (@available(macOS 15.0, *)) {
            options.mathMode = MTLMathModeSafe;
            const char *version = getenv("TOTK_TEST_MSL_VERSION");
            if (version && !strcmp(version, "2.3")) options.languageVersion = MTLLanguageVersion2_3;
            else if (version && !strcmp(version, "3.2")) options.languageVersion = MTLLanguageVersion3_2;
            else if (version) {
                fprintf(stderr, "TOTK_TEST_MSL_VERSION must be 2.3 or 3.2\n");
                return 2;
            }
        } else {
            fprintf(stderr, "this safe-math compilation probe requires macOS 15 or newer\n");
            return 1;
        }
        id<MTLLibrary> library = [device newLibraryWithSource:source options:options error:&error];
        if (!library) {
            fprintf(stderr, "compile MSL: %s\n", error.description.UTF8String);
            return 1;
        }
        if (library.functionNames.count != 1) {
            fprintf(stderr, "expected exactly one translated entry point\n");
            return 1;
        }
        nvn_refl parsed = {};
        if (argc == 3 || argc == 5) {
            NSData *data = [NSData dataWithContentsOfFile:@(argv[argc == 3 ? 2 : 4]) options:0 error:&error];
            NSDictionary *reflection = data ? [NSJSONSerialization JSONObjectWithData:data options:0 error:&error] : nil;
            if (![reflection isKindOfClass:[NSDictionary class]]) {
                fprintf(stderr, "read reflection: %s\n", error.description.UTF8String);
                return 1;
            }
            char message[1024];
            if (nvn_refl_from_json(reflection, &parsed, message, sizeof message)) {
                fprintf(stderr, "native reflection: %s\n", message);
                return 1;
            }
            if (parsed.nb) {
                id<MTLFunction> function = [library newFunctionWithName:library.functionNames[0]];
                id<MTLArgumentEncoder> encoder = [function newArgumentEncoderWithBufferIndex:0];
                if (!encoder || !encoder.encodedLength) {
                    fprintf(stderr, "cannot create Metal argument encoder for reflected buffer 0\n");
                    return 1;
                }
            }
            if (parsed.tg[0]) {
                id<MTLFunction> function = [library newFunctionWithName:library.functionNames[0]];
                if (![device newComputePipelineStateWithFunction:function error:&error]) {
                    fprintf(stderr, "create compute pipeline: %s\n", error.description.UTF8String);
                    return 1;
                }
            }
        }
        if (argc == 5 && !strcmp(argv[2], "compute")) {
            return ComputeProbe(device, library, &parsed, argv[3]);
        }
        if (argc >= 4) {
            NSString *vertexSource =
                @"#include <metal_stdlib>\nusing namespace metal;\n"
                 "vertex float4 probe_vertex(uint id [[vertex_id]]) {\n"
                 "float2 p[3] = {float2(-1,-1), float2(3,-1), float2(-1,3)};\n"
                 "return float4(p[id], 0, 1); }\n"
                 "struct probe_depth_out { float z [[depth(any)]]; };\n"
                 "fragment probe_depth_out probe_depth(float4 p [[position]]) {\n"
                 "return {0.125f + (p.x >= 1 ? 0.125f : 0) + (p.y >= 1 ? 0.5f : 0)}; }\n";
            id<MTLLibrary> vertices = [device newLibraryWithSource:vertexSource options:options error:&error];
            if (!vertices) {
                fprintf(stderr, "compile probe vertex: %s\n", error.description.UTF8String);
                return 1;
            }
            BOOL depthOutput = !strcmp(argv[2], "depth");
            MTLPixelFormat format;
            if (!strcmp(argv[2], "uint")) format = MTLPixelFormatRGBA32Uint;
            else if (!strcmp(argv[2], "sint")) format = MTLPixelFormatRGBA32Sint;
            else if (!strcmp(argv[2], "float") || depthOutput) format = MTLPixelFormatRGBA32Float;
            else return 2;
            MTLRenderPipelineDescriptor *descriptor = [MTLRenderPipelineDescriptor new];
            descriptor.vertexFunction = [vertices newFunctionWithName:@"probe_vertex"];
            descriptor.fragmentFunction = [library newFunctionWithName:library.functionNames[0]];
            descriptor.colorAttachments[0].pixelFormat = format;
            if (depthOutput) descriptor.depthAttachmentPixelFormat = MTLPixelFormatDepth32Float;
            id<MTLRenderPipelineState> pipeline =
                [device newRenderPipelineStateWithDescriptor:descriptor error:&error];
            if (!pipeline) {
                fprintf(stderr, "create pipeline: %s\n", error.description.UTF8String);
                return 1;
            }
            MTLTextureDescriptor *textureDescriptor =
                [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:format
                                                                 width:1 height:1 mipmapped:NO];
            textureDescriptor.usage = MTLTextureUsageRenderTarget;
            textureDescriptor.storageMode = MTLStorageModeShared;
            id<MTLTexture> target = [device newTextureWithDescriptor:textureDescriptor];
            id<MTLCommandQueue> queue = [device newCommandQueue];
            id<MTLCommandBuffer> commands = [queue commandBuffer];
            if (!target || !queue || !commands) {
                fprintf(stderr, "cannot allocate GPU probe resources\n");
                return 1;
            }
            MTLRenderPassDescriptor *pass = [MTLRenderPassDescriptor renderPassDescriptor];
            pass.colorAttachments[0].texture = target;
            pass.colorAttachments[0].loadAction = MTLLoadActionClear;
            pass.colorAttachments[0].storeAction = MTLStoreActionStore;
            id<MTLTexture> depthTarget = nil;
            id<MTLBuffer> depthReadback = nil;
            if (depthOutput) {
                MTLTextureDescriptor *d = [MTLTextureDescriptor
                    texture2DDescriptorWithPixelFormat:MTLPixelFormatDepth32Float
                    width:1 height:1 mipmapped:NO];
                d.usage = MTLTextureUsageRenderTarget;
                d.storageMode = MTLStorageModePrivate;
                depthTarget = [device newTextureWithDescriptor:d];
                depthReadback = [device newBufferWithLength:256 options:MTLResourceStorageModeShared];
                if (!depthTarget || !depthReadback) {
                    fprintf(stderr, "cannot allocate depth output fixture\n");
                    return 1;
                }
                pass.depthAttachment.texture = depthTarget;
                pass.depthAttachment.loadAction = MTLLoadActionClear;
                pass.depthAttachment.storeAction = MTLStoreActionStore;
            }
            id<MTLTexture> depth = nil;
            for (int i = 0; i < parsed.nb; i++) {
                if (parsed.b[i].kind != NVN_RB_TEX || parsed.b[i].dpf != NVN_DPF_DEPTH) continue;
                if (depth) continue;
                MTLTextureDescriptor *d = [MTLTextureDescriptor
                    texture2DDescriptorWithPixelFormat:MTLPixelFormatDepth32Float
                    width:2 height:2 mipmapped:NO];
                d.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
                d.storageMode = MTLStorageModePrivate;
                depth = [device newTextureWithDescriptor:d];
                MTLRenderPipelineDescriptor *p = [MTLRenderPipelineDescriptor new];
                p.vertexFunction = [vertices newFunctionWithName:@"probe_vertex"];
                p.fragmentFunction = [vertices newFunctionWithName:@"probe_depth"];
                p.depthAttachmentPixelFormat = MTLPixelFormatDepth32Float;
                id<MTLRenderPipelineState> state =
                    [device newRenderPipelineStateWithDescriptor:p error:&error];
                MTLDepthStencilDescriptor *ds = [MTLDepthStencilDescriptor new];
                ds.depthCompareFunction = MTLCompareFunctionAlways;
                ds.depthWriteEnabled = YES;
                MTLRenderPassDescriptor *dp = [MTLRenderPassDescriptor renderPassDescriptor];
                dp.depthAttachment.texture = depth;
                dp.depthAttachment.loadAction = MTLLoadActionClear;
                dp.depthAttachment.storeAction = MTLStoreActionStore;
                id<MTLRenderCommandEncoder> de =
                    depth && state ? [commands renderCommandEncoderWithDescriptor:dp] : nil;
                if (!de) {
                    fprintf(stderr, "create depth fixture: %s\n", error.description.UTF8String);
                    return 1;
                }
                [de setRenderPipelineState:state];
                [de setDepthStencilState:[device newDepthStencilStateWithDescriptor:ds]];
                [de drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
                [de endEncoding];
            }
            id<MTLRenderCommandEncoder> encoder = [commands renderCommandEncoderWithDescriptor:pass];
            if (!encoder) {
                fprintf(stderr, "cannot create GPU probe encoder\n");
                return 1;
            }
            [encoder setRenderPipelineState:pipeline];
            if (depthOutput) {
                MTLDepthStencilDescriptor *d = [MTLDepthStencilDescriptor new];
                d.depthCompareFunction = MTLCompareFunctionAlways;
                d.depthWriteEnabled = YES;
                [encoder setDepthStencilState:[device newDepthStencilStateWithDescriptor:d]];
            }
            NSMutableArray<id<MTLResource>> *resources = [NSMutableArray array];
            if (argc == 5) {
                id<MTLArgumentEncoder> arguments =
                    [descriptor.fragmentFunction newArgumentEncoderWithBufferIndex:parsed.abi];
                id<MTLBuffer> table = [device newBufferWithLength:arguments.encodedLength
                                                         options:MTLResourceStorageModeShared];
                if (!arguments || !table || !parsed.nb) {
                    fprintf(stderr, "expected argument-buffer GPU fixture bindings\n");
                    return 1;
                }
                [arguments setArgumentBuffer:table offset:0];
                [resources addObject:table];
                for (int i = 0; i < parsed.nb; i++) {
                    nvn_rbind binding = parsed.b[i];
                    if (binding.kind == NVN_RB_SAMP || binding.kind == NVN_RB_SAMPT) {
                        MTLSamplerDescriptor *s = [MTLSamplerDescriptor new];
                        s.supportArgumentBuffers = YES;
                        s.minFilter = s.magFilter = MTLSamplerMinMagFilterLinear;
                        s.sAddressMode = s.tAddressMode = MTLSamplerAddressModeClampToEdge;
                        s.compareFunction = MTLCompareFunctionLessEqual;
                        id<MTLSamplerState> sampler = [device newSamplerStateWithDescriptor:s];
                        if (!sampler) {
                            fprintf(stderr, "cannot allocate GPU fixture sampler\n");
                            return 1;
                        }
                        [arguments setSamplerState:sampler atIndex:binding.aid];
                        continue;
                    }
                    if (binding.kind == NVN_RB_TEX) {
                        id<MTLTexture> texture = depth;
                        if (binding.dpf == NVN_DPF_FLOAT &&
                            binding.dtt == MTLTextureType2DArray) {
                            MTLTextureDescriptor *d = [MTLTextureDescriptor
                                texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA32Float
                                width:4 height:4 mipmapped:NO];
                            d.textureType = MTLTextureType2DArray;
                            d.arrayLength = 2;
                            d.storageMode = MTLStorageModeShared;
                            d.usage = MTLTextureUsageShaderRead;
                            texture = [device newTextureWithDescriptor:d];
                            for (NSUInteger layer = 0; layer < 2; ++layer) {
                                float values[4][4][4];
                                for (NSUInteger y = 0; y < 4; ++y) {
                                    for (NSUInteger x = 0; x < 4; ++x) {
                                        values[y][x][0] = (float)(1 + x + 4 * y + 16 * layer);
                                        values[y][x][1] = 0.25f;
                                        values[y][x][2] = 0.75f;
                                        values[y][x][3] = 1.0f;
                                    }
                                }
                                [texture replaceRegion:MTLRegionMake2D(0, 0, 4, 4) mipmapLevel:0
                                                 slice:layer withBytes:values
                                           bytesPerRow:sizeof(values[0])
                                         bytesPerImage:sizeof(values)];
                            }
                        } else if (binding.dpf == NVN_DPF_FLOAT) {
                            MTLTextureDescriptor *d = [MTLTextureDescriptor
                                texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA32Float
                                width:2 height:2 mipmapped:NO];
                            d.storageMode = MTLStorageModeShared;
                            d.usage = MTLTextureUsageShaderRead;
                            texture = [device newTextureWithDescriptor:d];
                            const float values[4][4] = {
                                {0.5, 0.25, 0.75, 1}, {0.5, 0.25, 0.75, 1},
                                {0.5, 0.25, 0.75, 1}, {0.5, 0.25, 0.75, 1}};
                            [texture replaceRegion:MTLRegionMake2D(0, 0, 2, 2) mipmapLevel:0
                                         withBytes:values bytesPerRow:sizeof(values[0]) * 2];
                        }
                        if (!texture || binding.dtt != texture.textureType) {
                            fprintf(stderr, "unexpected GPU texture fixture binding\n");
                            return 1;
                        }
                        [resources addObject:texture];
                        [arguments setTexture:texture atIndex:binding.aid];
                        [encoder useResource:texture usage:MTLResourceUsageRead
                                      stages:MTLRenderStageFragment];
                        continue;
                    }
                    id<MTLBuffer> buffer = FixtureBuffer(device, &parsed, binding);
                    if (!buffer) return 1;
                    [resources addObject:buffer];
                    [arguments setBuffer:buffer offset:0 atIndex:binding.aid];
                    MTLResourceUsage usage = MTLResourceUsageRead;
                    if (binding.rw) usage |= MTLResourceUsageWrite;
                    [encoder useResource:buffer usage:usage stages:MTLRenderStageFragment];
                }
                [encoder setFragmentBuffer:table offset:0 atIndex:parsed.abi];
            }
            [encoder drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
            [encoder endEncoding];
            if (depthOutput) {
                id<MTLBlitCommandEncoder> blit = [commands blitCommandEncoder];
                [blit copyFromTexture:depthTarget sourceSlice:0 sourceLevel:0
                        sourceOrigin:MTLOriginMake(0, 0, 0) sourceSize:MTLSizeMake(1, 1, 1)
                            toBuffer:depthReadback destinationOffset:0
               destinationBytesPerRow:256 destinationBytesPerImage:256];
                [blit endEncoding];
            }
            [commands commit];
            [commands waitUntilCompleted];
            if (commands.status != MTLCommandBufferStatusCompleted) {
                fprintf(stderr, "GPU probe failed: %s\n", commands.error.description.UTF8String);
                return 1;
            }
            uint32_t pixel[4] = {};
            if (depthOutput) {
                memcpy(pixel, depthReadback.contents, sizeof pixel[0]);
            } else {
                [target getBytes:pixel bytesPerRow:sizeof pixel fromRegion:MTLRegionMake2D(0, 0, 1, 1)
                     mipmapLevel:0];
            }
            NSArray<NSString *> *expectedChannels = [@(argv[3]) componentsSeparatedByString:@","];
            if (expectedChannels.count != 1 && expectedChannels.count != 4) return 2;
            for (NSUInteger i = 0; i < expectedChannels.count; i++) {
                const char *text = expectedChannels[i].UTF8String;
                char *end = NULL;
                unsigned long expected = strtoul(text, &end, 16);
                if (!text[0] || *end || expected > UINT32_MAX || pixel[i] != expected) {
                    fprintf(stderr, "GPU channel %lu: %08x, expected %s\n",
                            (unsigned long)i, pixel[i], text);
                    return 1;
                }
            }
        }
    }
    return 0;
}
