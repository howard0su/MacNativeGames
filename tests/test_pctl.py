"""Game-data-free CMIF regression for the local parental-control service."""
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

PROBE = r'''
#include "runtime.h"
#include "ipc.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

pthread_key_t runtime_keys[8];
static IpcSession *sessions[32];
static unsigned count;
static int domain_reply;
static unsigned out_objects;
_Noreturn void runtime_fail(const char *message) {
    fprintf(stderr, "%s\n", message);
    exit(1);
}
void runtime_trace_guest(uint64_t fp, uint64_t lr) { (void)fp; (void)lr; }
uint32_t runtime_handle_new(int session, void *obj) {
    assert(session == 1 && count + 1 < 32);
    sessions[++count] = obj;
    return count;
}
void *runtime_handle_get(uint32_t handle, int session) {
    assert(session == 1);
    return handle && handle <= count ? sessions[handle] : NULL;
}

static void request(uint32_t *m, unsigned cmd, const void *data, unsigned size, int pid) {
    domain_reply = 0;
    memset(m, 0, 0x100);
    m[0] = 6;  /* SDK 7 RequestWithContext */
    unsigned words = 2;
    if (pid) {
        m[1] = 1u << 31;
        m[2] = 1;
        m[3] = 42;
        words = 5;
    }
    unsigned payload = (words + 3) & ~3u;
    m[payload] = 0x49434653;
    m[payload + 2] = cmd;
    if (size) memcpy(m + payload + 4, data, size);
    m[1] |= payload + 4 + (size + 3) / 4 - words;
}
static void domain_request(uint32_t *m, uint32_t id, unsigned cmd,
                           const void *data, unsigned size, int pid) {
    request(m, cmd, data, size, pid);
    unsigned payload = pid ? 8 : 4;
    memmove(m + payload + 4, m + payload, 16 + size);
    memset(m + payload, 0, 16);
    m[payload] = 1 | ((16 + size) << 16);
    m[payload + 1] = id;
    m[1] += 4;
    domain_reply = 1;
}
static uint32_t *send(uint32_t handle, uint32_t *m) {
    uint64_t regs[31] = {0};
    regs[0] = (uintptr_t)m;
    regs[1] = 0x100;
    regs[2] = handle;
    assert(runtime_ipc_svc(0x22, regs) == 1 && regs[0] == 0);
    unsigned special = m[1] >> 31;
    unsigned offset = 2;
    if (special) offset += 1 + ((m[2] >> 1) & 15) + ((m[2] >> 5) & 15);
    offset = (offset + 3) & ~3u;
    if (domain_reply) {
        out_objects = m[offset];
        assert(!(m[1] >> 31));
        offset += 4;
    }
    assert(m[offset] == 0x4f434653 && m[offset + 2] == 0);
    return m + offset + 4;
}
static uint32_t moved(const uint32_t *m) {
    assert(m[1] >> 31 && m[2] == (1u << 5));
    assert(m[3] && m[3] <= count);
    return m[3];
}
static void check_bool(uint32_t handle, unsigned cmd, uint8_t expected) {
    uint32_t m[64];
    request(m, cmd, NULL, 0, 0);
    assert(*(uint8_t *)send(handle, m) == expected);
    IpcRequest rq = {.cmd = cmd};
    IpcResponse rs = {0};
    sessions[handle]->handler(sessions[handle], &rq, &rs);
    assert(rs.result == 0 && rs.data_size == 1 && rs.data[0] == expected);
}
static void domain_checks(uint32_t factory, const char *mode) {
    uint32_t m[64];
    request(m, 0, NULL, 0, 0); m[0] = 7;
    uint32_t root = *send(factory, m);
    assert(root != 0);
    request(m, 3, NULL, 0, 0); m[0] = 7;
    assert(*(uint16_t *)send(factory, m) == 0x8000);
    request(m, 4, NULL, 0, 0); m[0] = 7;
    send(factory, m);
    uint32_t clone = moved(m);
    runtime_ipc_session_closed(sessions[factory]);
    factory = clone;  /* A cloned handle must keep the table and objects alive. */
    for (unsigned create = 0; create <= 1; create++) {
        uint64_t pid = 0;
        domain_request(m, root, create, &pid, sizeof pid, 1);
        uint32_t service = *send(factory, m);
        assert(out_objects == 1 && service != 0 && service != root);
        domain_request(m, service, 1, NULL, 0, 0); send(factory, m);
        if (!strcmp(mode, "domain-invalid-object")) {
            domain_request(m, 63, 1065, NULL, 0, 0); send(factory, m); exit(2);
        }
        if (!strcmp(mode, "domain-short-header")) {
            domain_request(m, service, 1065, NULL, 0, 0); m[1] -= 8;
            send(factory, m); exit(2);
        }
        if (!strcmp(mode, "domain-short-payload")) {
            domain_request(m, service, 1065, NULL, 0, 0); m[4] = 1 | (15u << 16);
            send(factory, m); exit(2);
        }
        if (!strcmp(mode, "domain-input-object")) {
            domain_request(m, service, 1065, NULL, 0, 0); m[4] |= 1u << 8;
            send(factory, m); exit(2);
        }
        if (!strcmp(mode, "domain-unknown-command")) {
            domain_request(m, service, 9999, NULL, 0, 0); send(factory, m); exit(2);
        }
        const unsigned cmds[] = {1006, 1013, 1031, 1062, 1064, 1065};
        for (unsigned i = 0; i < sizeof cmds / sizeof *cmds; i++) {
            domain_request(m, service, cmds[i], NULL, 0, 0);
            uint32_t *data = send(factory, m);
            assert(out_objects == 0);
            if (cmds[i] == 1065) assert(*(uint8_t *)data == 1);
            else assert(*data == 0);
        }
        domain_request(m, service, 0, NULL, 0, 0);
        m[4] = 2; send(factory, m);
        if (!strcmp(mode, "domain-closed-object")) {
            domain_request(m, service, 1065, NULL, 0, 0); send(factory, m); exit(2);
        }
    }
    /* Repeated create/close must reuse slots instead of exhausting the domain. */
    for (unsigned i = 0; i < 80; i++) {
        uint64_t pid = 0;
        domain_request(m, root, 0, &pid, sizeof pid, 1);
        uint32_t service = *send(factory, m);
        domain_request(m, service, 0, NULL, 0, 0);
        m[4] = 2; send(factory, m);
    }
    runtime_ipc_session_closed(sessions[factory]);
}
static void closed(void *state) { ++*(unsigned *)state; }
static void lifetime_checks(void) {
    unsigned closes = 0;
    uint32_t m[64], handle = runtime_ipc_new_session("lifetime", NULL, &closes, closed);
    request(m, 0, NULL, 0, 0); m[0] = 5; send(handle, m);
    request(m, 2, NULL, 0, 0); m[0] = 5; send(handle, m);
    uint32_t clone = moved(m);
    runtime_ipc_session_closed(sessions[handle]);
    assert(closes == 0);
    runtime_ipc_session_closed(sessions[clone]);
    assert(closes == 1);
    handle = runtime_ipc_new_session("lifetime", NULL, &closes, closed);
    request(m, 0, NULL, 0, 0); m[0] = 5;
    uint32_t root = *send(handle, m);
    domain_request(m, root, 0, NULL, 0, 0); m[4] = 2; send(handle, m);
    assert(closes == 2);
    runtime_ipc_session_closed(sessions[handle]);
    assert(closes == 2);
}
int main(int argc, char **argv) {
    uint64_t regs[31] = {0};
    char port[13] = "sm:";
    regs[1] = (uintptr_t)port;
    assert(runtime_ipc_svc(0x1f, regs) == 1 && regs[0] == 0);
    uint32_t sm = (uint32_t)regs[1], m[64];
    char name[8] = {0};
    const char *service_name = argc > 2 ? argv[2] : "pctl";
    assert(strlen(service_name) <= sizeof name);
    memcpy(name, service_name, strlen(service_name));
    request(m, 1, name, sizeof name, 0);
    send(sm, m);
    uint32_t factory = moved(m);
    const char *mode = argc > 1 ? argv[1] : "";
    if (!strncmp(mode, "domain", 6)) {
        domain_checks(factory, mode);
        runtime_ipc_session_closed(sessions[sm]);
        lifetime_checks();
        puts("PASS domain"); return 0;
    }
    if (argc > 1 && !strcmp(argv[1], "unknown-factory")) {
        request(m, 99, NULL, 0, 0); send(factory, m); return 2;
    }
    if (argc > 1 && !strcmp(argv[1], "short-create")) {
        request(m, 1, NULL, 0, 1); send(factory, m); return 2;
    }
    for (unsigned create = 0; create <= 1; create++) {
        uint64_t pid = 0;
        request(m, create, &pid, sizeof pid, 1);
        if (argc > 1 && !strcmp(argv[1], "missing-pid"))
            request(m, create, &pid, sizeof pid, 0);
        send(factory, m);
        uint32_t service = moved(m);
        assert(service != factory);
        if (argc > 1 && !strcmp(argv[1], "unknown-service")) {
            request(m, 1063, NULL, 0, 0); send(service, m); return 2;
        }
        const unsigned no_output[] = {1, 1013, 1064};
        for (unsigned i = 0; i < sizeof no_output / sizeof *no_output; i++) {
            request(m, no_output[i], NULL, 0, 0); send(service, m);
            IpcRequest rq = {.cmd = no_output[i]};
            IpcResponse rs = {0};
            sessions[service]->handler(sessions[service], &rq, &rs);
            assert(rs.result == 0 && rs.data_size == 0 && !rs.ncopy && !rs.nmove);
        }
        check_bool(service, 1006, 0);
        check_bool(service, 1031, 0);
        check_bool(service, 1062, 0);
        check_bool(service, 1065, 1);
        runtime_ipc_session_closed(sessions[service]);
    }
    runtime_ipc_session_closed(sessions[factory]);
    runtime_ipc_session_closed(sessions[sm]);
    puts("PASS");
    return 0;
}
'''


@unittest.skipUnless(shutil.which('xcrun'), 'needs the Xcode command line tools')
class ParentalControlTests(unittest.TestCase):
    def test_cmif_sessions_permissions_and_failures(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / 'probe.c'
            source.write_text(PROBE)
            probe = Path(tmp) / 'probe'
            runtime = ROOT / 'games/totk/runtime'
            build = subprocess.run(
                ['xcrun', 'clang', '-std=c11', '-Wall', '-Wextra', '-Werror',
                 '-fsanitize=undefined', '-fno-sanitize-recover=all', '-I', str(runtime),
                 str(source), str(runtime / 'ipc.c'), str(runtime / 'pctl_service.c'),
                 '-o', str(probe)], capture_output=True, text=True)
            self.assertEqual(build.returncode, 0, build.stderr[-4000:])
            for service in ('pctl', 'pctl:s', 'pctl:a', 'pctl:r'):
                for mode in ('', 'domain'):
                    with self.subTest(service=service, mode=mode):
                        run = subprocess.run([str(probe), mode, service],
                                             capture_output=True, text=True, timeout=30)
                        self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
                        self.assertIn('PASS', run.stdout)
            for mode, error in (
                ('unknown-factory', 'command 99 not implemented'),
                ('unknown-service', 'command 1063 not implemented'),
                ('short-create', 'argument out of range'),
                ('missing-pid', 'needs a PID descriptor'),
                ('domain-invalid-object', 'invalid domain object'),
                ('domain-closed-object', 'invalid domain object'),
                ('domain-short-header', 'truncated CMIF header'),
                ('domain-short-payload', 'invalid domain payload size'),
                ('domain-input-object', 'input objects not implemented'),
                ('domain-unknown-command', 'command 9999 not implemented'),
            ):
                with self.subTest(mode=mode):
                    run = subprocess.run([str(probe), mode], capture_output=True, text=True, timeout=30)
                    self.assertEqual(run.returncode, 1, run.stdout + run.stderr)
                    self.assertIn(error, run.stderr)
