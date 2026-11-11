// Minimal CUPTI activity tracer, preloaded into a CUDA process.
//
// `nsys` collects but its qdstrm importer aborts against this driver (595.112 /
// CUDA 13.2) because the packaged nsight-systems is the CUDA 12.9 build, and
// `ncu` cannot read performance counters here (RmProfilingAdminOnly=1). CUPTI
// *activity* tracing needs no such permission, so this prints per-kernel and
// per-copy device time straight to a CSV:
//
//   CUPTI_TRACE_OUT=/tmp/trace.csv LD_PRELOAD=/path/libcupti_trace.so <binary>
//
// Columns: kind,start_ns,end_ns,dur_ns,stream,bytes_or_0,name
#include <cuda.h>
#include <cupti.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define BUF_SIZE (8 * 1024 * 1024)
#define BUF_ALIGN 8

static FILE *g_out = NULL;
static unsigned long long g_records = 0;

static const char *kind_name(CUpti_ActivityKind k) {
    switch (k) {
        case CUPTI_ACTIVITY_KIND_KERNEL: return "kernel";
        case CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL: return "kernel";
        case CUPTI_ACTIVITY_KIND_MEMCPY: return "memcpy";
        case CUPTI_ACTIVITY_KIND_MEMSET: return "memset";
        default: return NULL;
    }
}

static void CUPTIAPI on_buffer_requested(uint8_t **buffer, size_t *size, size_t *max_num_records) {
    *buffer = (uint8_t *)malloc(BUF_SIZE + BUF_ALIGN);
    *size = BUF_SIZE;
    *max_num_records = 0;
}

static void CUPTIAPI on_buffer_completed(CUcontext ctx, uint32_t stream_id, uint8_t *buffer,
                                         size_t size, size_t valid_size) {
    (void)ctx;
    (void)stream_id;
    (void)size;
    CUpti_Activity *rec = NULL;
    while (cuptiActivityGetNextRecord(buffer, valid_size, &rec) == CUPTI_SUCCESS) {
        g_records++;
        const char *kind = kind_name(rec->kind);
        if (!kind) {
            continue;
        }
        if (rec->kind == CUPTI_ACTIVITY_KIND_KERNEL || rec->kind == CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL) {
            CUpti_ActivityKernel9 *k = (CUpti_ActivityKernel9 *)rec;
            fprintf(g_out, "%s,%llu,%llu,%llu,%u,0,%s\n", kind,
                    (unsigned long long)k->start, (unsigned long long)k->end,
                    (unsigned long long)(k->end - k->start), k->streamId,
                    k->name ? k->name : "?");
        } else if (rec->kind == CUPTI_ACTIVITY_KIND_MEMCPY) {
            CUpti_ActivityMemcpy6 *m = (CUpti_ActivityMemcpy6 *)rec;
            fprintf(g_out, "%s,%llu,%llu,%llu,%u,%llu,kind=%u\n", kind,
                    (unsigned long long)m->start, (unsigned long long)m->end,
                    (unsigned long long)(m->end - m->start), m->streamId,
                    (unsigned long long)m->bytes, (unsigned)m->copyKind);
        } else {
            CUpti_ActivityMemset4 *s = (CUpti_ActivityMemset4 *)rec;
            fprintf(g_out, "%s,%llu,%llu,%llu,%u,%llu,\n", kind,
                    (unsigned long long)s->start, (unsigned long long)s->end,
                    (unsigned long long)(s->end - s->start), s->streamId,
                    (unsigned long long)s->bytes);
        }
    }
    fflush(g_out);
    free(buffer);
}

__attribute__((constructor)) static void trace_init(void) {
    const char *path = getenv("CUPTI_TRACE_OUT");
    if (!path || !*path) {
        return;
    }
    // Several processes in one run inherit LD_PRELOAD, so never share a path.
    char per_pid[4096];
    snprintf(per_pid, sizeof(per_pid), "%s.%d", path, (int)getpid());
    g_out = fopen(per_pid, "w");
    if (!g_out) {
        return;
    }
    fprintf(g_out, "kind,start_ns,end_ns,dur_ns,stream,bytes,name\n");
    uint32_t ver = 0;
    cuptiGetVersion(&ver);
    fprintf(stderr, "[cupti] version=%u\n", ver);
    CUptiResult rc;
    rc = cuptiActivityRegisterCallbacks(on_buffer_requested, on_buffer_completed);
    fprintf(stderr, "[cupti] register rc=%d\n", (int)rc);
    rc = cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL);
    fprintf(stderr, "[cupti] enable kernel rc=%d\n", (int)rc);
    rc = cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMCPY);
    fprintf(stderr, "[cupti] enable memcpy rc=%d\n", (int)rc);
}

__attribute__((destructor)) static void trace_fini(void) {
    if (!g_out) {
        return;
    }
    CUptiResult rc = cuptiActivityFlushAll(1);
    fprintf(stderr, "[cupti] flush rc=%d records=%llu\n", (int)rc, (unsigned long long)g_records);
    fflush(g_out);
    fclose(g_out);
    g_out = NULL;
}
