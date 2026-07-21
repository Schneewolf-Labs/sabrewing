/* Grouped-int4 (fmt=4) CUDA kernel oracle (#334).
 *
 * Feeds random offset-binary nibble weights + [O, ng] group scales through
 * grouped_hidden_g4_dual / grouped_down_g4 and checks against a CPU reference
 * that replicates matmul_i4_grouped's semantics (value = nibble - 8, per-group
 * partial dot x scale). Covers gs=64, a non-divisible tail group, and a
 * per-row (gs=0) member riding in the same launch — the fmt=2-compat case.
 *
 * The device buffers get the same XOR 0x88 offset->signed conversion the
 * upload path applies, so the kernels are exercised exactly as deployed.
 *
 * Build: nvcc -O2 -std=c++17 -arch=native tests/test_grouped_g4_cuda.cu -o tests/test_grouped_g4
 */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cuda_runtime.h>

#include "../backend_cuda.cu"

static void cpu_gemv_g4(const uint8_t *q,const float *sc,int K,int O,int gs,
                        const float *x,float *y){
    int rb=(K+1)/2, ng=gs>0?(K+gs-1)/gs:1, egs=gs>0?gs:K;
    for(int o=0;o<O;o++){
        const uint8_t *row=q+(size_t)o*rb; const float *scl=sc+(size_t)o*ng;
        double a=0;
        for(int g=0; g*egs<K; g++){
            int base=g*egs, glen=egs; if(base+glen>K) glen=K-base;
            double p=0;
            for(int i=base;i<base+glen;i++){
                uint8_t v=row[i>>1]; int n=(i&1)?(v>>4):(v&15);
                p+=(double)x[i]*(n-8);
            }
            a+=p*scl[g];
        }
        y[o]=(float)a;
    }
}

int main(void){
    srand(7);
    const int D=200, I=96, gs=64;            /* tail group: 200 % 64 = 8 */
    const int COUNT=3;                       /* expert 0,1: fmt4 gs=64; expert 2: per-row (gs=0) */
    const int rbD=(D+1)/2, rbI=(I+1)/2;
    const int ngD=(D+gs-1)/gs, ngI=(I+gs-1)/gs;
    int trials=50, bad=0;
    for(int t=0;t<trials;t++){
        GroupDesc host[COUNT]; float *xs; cudaMallocManaged(&xs,(size_t)COUNT*D*4);
        float *gate,*up,*y; cudaMallocManaged(&gate,(size_t)COUNT*I*4);
        cudaMallocManaged(&up,(size_t)COUNT*I*4); cudaMallocManaged(&y,(size_t)COUNT*D*4);
        uint8_t *qg[COUNT],*qu[COUNT],*qd[COUNT]; float *sg[COUNT],*su[COUNT],*sd[COUNT];
        uint8_t *hg[COUNT],*hu[COUNT],*hd[COUNT]; float *hgs[COUNT],*hus[COUNT],*hds[COUNT];
        for(int c=0;c<COUNT;c++){
            int cgs = c==2 ? 0 : gs;
            int cngD = cgs? ngD:1, cngI = cgs? ngI:1;
            hg[c]=(uint8_t*)malloc((size_t)I*rbD); hu[c]=(uint8_t*)malloc((size_t)I*rbD);
            hd[c]=(uint8_t*)malloc((size_t)D*rbI);
            hgs[c]=(float*)malloc((size_t)I*cngD*4); hus[c]=(float*)malloc((size_t)I*cngD*4);
            hds[c]=(float*)malloc((size_t)D*cngI*4);
            for(size_t i=0;i<(size_t)I*rbD;i++){ hg[c][i]=rand()&255; hu[c][i]=rand()&255; }
            for(size_t i=0;i<(size_t)D*rbI;i++) hd[c][i]=rand()&255;
            for(size_t i=0;i<(size_t)I*cngD;i++){ hgs[c][i]=.01f+.05f*(rand()/(float)RAND_MAX);
                                                  hus[c][i]=.01f+.05f*(rand()/(float)RAND_MAX); }
            for(size_t i=0;i<(size_t)D*cngI;i++) hds[c][i]=.01f+.05f*(rand()/(float)RAND_MAX);
            cudaMalloc(&qg[c],(size_t)I*rbD); cudaMalloc(&qu[c],(size_t)I*rbD); cudaMalloc(&qd[c],(size_t)D*rbI);
            cudaMalloc(&sg[c],(size_t)I*cngD*4); cudaMalloc(&su[c],(size_t)I*cngD*4); cudaMalloc(&sd[c],(size_t)D*cngI*4);
            cudaMemcpy(qg[c],hg[c],(size_t)I*rbD,cudaMemcpyHostToDevice);
            cudaMemcpy(qu[c],hu[c],(size_t)I*rbD,cudaMemcpyHostToDevice);
            cudaMemcpy(qd[c],hd[c],(size_t)D*rbI,cudaMemcpyHostToDevice);
            offset_to_signed_s4<<<64,256>>>(qg[c],(size_t)I*rbD);
            offset_to_signed_s4<<<64,256>>>(qu[c],(size_t)I*rbD);
            offset_to_signed_s4<<<64,256>>>(qd[c],(size_t)D*rbI);
            cudaMemcpy(sg[c],hgs[c],(size_t)I*cngD*4,cudaMemcpyHostToDevice);
            cudaMemcpy(su[c],hus[c],(size_t)I*cngD*4,cudaMemcpyHostToDevice);
            cudaMemcpy(sd[c],hds[c],(size_t)D*cngI*4,cudaMemcpyHostToDevice);
            host[c]={qg[c],qu[c],qd[c],sg[c],su[c],sd[c],4,4,4,1,c,cgs,cgs,cgs};
        }
        for(size_t i=0;i<(size_t)COUNT*D;i++) xs[i]=(rand()/(float)RAND_MAX-.5f)*2.f;
        GroupDesc *ddesc; cudaMalloc(&ddesc,sizeof(host));
        cudaMemcpy(ddesc,host,sizeof(host),cudaMemcpyHostToDevice);
        dim3 hgd((unsigned)I,1,(unsigned)COUNT),ogd((unsigned)D,1,(unsigned)COUNT);
        grouped_hidden_g4_dual<<<hgd,256>>>(gate,up,xs,ddesc,I,D);
        grouped_down_g4<<<ogd,256>>>(y,gate,ddesc,D,I);
        if(cudaDeviceSynchronize()!=cudaSuccess){ printf("FAIL cuda\n"); return 1; }
        for(int c=0;c<COUNT;c++){
            int cgs=c==2?0:gs;
            float rg[512],ru[512],ry[512];
            cpu_gemv_g4(hg[c],hgs[c],D,I,cgs,xs+(size_t)c*D,rg);
            cpu_gemv_g4(hu[c],hus[c],D,I,cgs,xs+(size_t)c*D,ru);
            for(int o=0;o<I;o++){
                if(fabsf(gate[(size_t)c*I+o]-rg[o])>1e-3f*(fabsf(rg[o])+1e-3f)||
                   fabsf(up[(size_t)c*I+o]-ru[o])>1e-3f*(fabsf(ru[o])+1e-3f)) bad++;
            }
            cpu_gemv_g4(hd[c],hds[c],I,D,cgs,(float*)gate+(size_t)c*I,ry);
            for(int o=0;o<D;o++)
                if(fabsf(y[(size_t)c*D+o]-ry[o])>1e-3f*(fabsf(ry[o])+1e-3f)) bad++;
        }
        for(int c=0;c<COUNT;c++){ cudaFree(qg[c]);cudaFree(qu[c]);cudaFree(qd[c]);
            cudaFree(sg[c]);cudaFree(su[c]);cudaFree(sd[c]);
            free(hg[c]);free(hu[c]);free(hd[c]);free(hgs[c]);free(hus[c]);free(hds[c]); }
        cudaFree(ddesc);cudaFree(xs);cudaFree(gate);cudaFree(up);cudaFree(y);
    }
    printf("grouped-g4 oracle: %d trials x %d experts (gs=64 + tail + per-row member), %d mismatches\n",
           trials,COUNT,bad);
    if(bad){ printf("FAIL\n"); return 1; }
    printf("OK\n"); return 0;
}
