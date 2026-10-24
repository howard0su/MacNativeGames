/* Local profile has no configured parental restrictions. CMIF command IDs and payloads follow
   libnx services/pctl.c and switchbrew's Parental Control services. This does not enable Labo VR. */
#include "ipc.h"
#include "runtime.h"
#include <stdio.h>

static void pctl_service(IpcSession *s, const IpcRequest *rq, IpcResponse *rs) {
    switch (rq->cmd) {
    case 1:       /* Initialize */
    case 1013:    /* ConfirmStereoVisionPermission: no parental restriction to confirm */
    case 1064:    /* ResetConfirmedStereoVisionPermission: no restriction or PIN state */
        return;
    case 1006:    /* IsRestrictionTemporaryUnlocked */
    case 1031:    /* IsRestrictionEnabled */
    case 1062: {  /* GetStereoVisionRestriction */
        uint8_t restricted = 0;
        IPC_OUT(rs, restricted);
        return;
    }
    case 1065: {  /* IsStereoVisionPermitted: permission, not display capability */
        uint8_t permitted = 1;
        IPC_OUT(rs, permitted);
        return;
    }
    default: ipc_unimplemented(s, rq);
    }
}

static void pctl_factory(IpcSession *s, const IpcRequest *rq, IpcResponse *rs) {
    switch (rq->cmd) {
    case 0:       /* CreateService */
    case 1:       /* CreateServiceWithoutInitialize */
        (void)IPC_ARG(rq, uint64_t, 0);   /* PID placeholder; the real PID is in the HIPC header. */
        if (!rq->has_pid) runtime_fail("pctl CreateService needs a PID descriptor");
        ipc_out_session(s, rs, "pctl:service", pctl_service, NULL, NULL);
        fprintf(stderr, "[totk-pctl] local profile: parental restrictions disabled (Labo VR remains unsupported)\n");
        return;
    default: ipc_unimplemented(s, rq);
    }
}

__attribute__((constructor)) static void register_pctl(void) {
    runtime_ipc_register("pctl", pctl_factory, NULL, NULL);
    runtime_ipc_register("pctl:s", pctl_factory, NULL, NULL);
    runtime_ipc_register("pctl:a", pctl_factory, NULL, NULL);
    runtime_ipc_register("pctl:r", pctl_factory, NULL, NULL);
}
