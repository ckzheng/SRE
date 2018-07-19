#ifndef BLSTM_FORWARD_H
#define BLSTM_FORWARD_H
#include "base/log_message.h"
#include "am/xnn_sse.h"
#include <algorithm>
namespace idec{
    namespace blstm_forward{

        // the definition of the matrix 
        struct FloatMatrix {
            size_t num_cols;
            size_t num_rows;
            float  *data;
            size_t col_stride;
        };

        // col view of the matrix
        static void ColView(FloatMatrix    *mat, size_t start_col, size_t num_cols, FloatMatrix *out_view);
        // row&col view of the matrix
        static void ColRowView(FloatMatrix *mat, size_t start_col, size_t num_cols, size_t start_row, size_t num_rows, FloatMatrix *out_view);

        // operations need for blstm computation 
        static void PlusMatTMat(FloatMatrix*src_dst, FloatMatrix *Mt, FloatMatrix*V);// matrix-matrix multiplication
        static void Plusv(FloatMatrix*src_dst, FloatMatrix*v);
        static void ScalePlusvElemProdv(FloatMatrix*src_dst, const float scale, FloatMatrix *X, FloatMatrix *Y);        // dot-product
        static void SetZero(FloatMatrix *Ofw);
        static void Sigmoid(FloatMatrix *in, FloatMatrix *out);
        static void Tanh(FloatMatrix *in, FloatMatrix *out);
        static void FloorAndCeiling(FloatMatrix *in, FloatMatrix *out, float floor, float ceiling);
        static void SetValue(FloatMatrix *src, FloatMatrix *dst);


        void BlstmForwardProp(
            // weight & bias
            FloatMatrix *W_fw,
            FloatMatrix *R_fw,
            FloatMatrix *b_fw,
            FloatMatrix *pfw,


            FloatMatrix *W_bw,
            FloatMatrix *R_bw,
            FloatMatrix *b_bw,
            FloatMatrix *p_bw,

            // input & output 
            FloatMatrix *v,
            FloatMatrix *u,

            // intermediate states
            FloatMatrix *O_fw,
            FloatMatrix *c_fw,
            FloatMatrix *c_fw_nl,
            FloatMatrix *buf_ufw,
            FloatMatrix *buf_cfw,
            FloatMatrix *O_bw,
            FloatMatrix *c_bw,
            FloatMatrix *c_bw_nl,
            size_t wstride,
            size_t window_shift,
            bool   is_bidirectional) {

            // Step 1. activation calculated on non-recurrent input
            SetZero(O_fw);
            PlusMatTMat(O_fw, W_fw, v);
            Plusv(O_fw, b_fw);


            FloatMatrix O_fw_col;
            FloatMatrix u_fw;
            FloatMatrix input_gate;
            FloatMatrix forget_gate;
            FloatMatrix output_gate;
            FloatMatrix input_peephole;
            FloatMatrix forget_peephole;
            FloatMatrix output_peephole;

            for (size_t t = 0; t < v->num_cols; ++t) {
                ColView(O_fw, t, 1, &O_fw_col);
                if (t == 0) {
                    PlusMatTMat(&O_fw_col, R_fw, buf_ufw);
                }
                else {
                    ColRowView(u, t - 1, 1, 0, wstride, &u_fw);
                    PlusMatTMat(&O_fw_col, R_fw, &u_fw);
                }

                // peephole & non-linear for input gate
                ColRowView(O_fw, t, 1, wstride, wstride, &input_gate);
                ColView(pfw, 0, 1, &input_peephole);
                if (t != 0) {
                    ScalePlusvElemProdv(&input_gate, 1.0f, c_fw, &input_peephole);
                }
                else {
                    ScalePlusvElemProdv(&input_gate, 1.0f, buf_cfw, &input_peephole);
                }
                Sigmoid(&input_gate, &input_gate);


                // peephole & non-linear for forget gate
                ColRowView(O_fw, t, 1, wstride * 2, wstride, &forget_gate);
                ColView(pfw, 1, 1, &forget_peephole);
                if (t != 0) {
                    ScalePlusvElemProdv(&forget_gate, 1.0f, c_fw, &forget_peephole);
                }
                else {
                    ScalePlusvElemProdv(&forget_gate, 1.0f, buf_cfw, &forget_peephole);
                }
                Sigmoid(&forget_gate, &forget_gate);


                // non-linear on node input
                FloatMatrix ni;
                ColRowView(O_fw, t, 1, 0, wstride, &ni); // select node input
                Tanh(&ni, &ni);

                // Step 2(e). cell state:
                // forget gate
                if (t != 0) {
                    ScalePlusvElemProdv(c_fw, 0, c_fw, &forget_gate);
                }
                else {
                    ScalePlusvElemProdv(c_fw, 0, buf_cfw, &forget_gate);
                }
                // input gate
                ScalePlusvElemProdv(c_fw, 1.0f, &ni, &input_gate);
                FloorAndCeiling(c_fw, c_fw, -50, 50);


                // Step 2(g). non-linear cell state
                // pre-condition, make sure they have the same size
                Tanh(c_fw, c_fw_nl);

                // Step 2(f). peephole for og
                ColRowView(O_fw, t, 1, wstride * 3, wstride, &output_gate);
                ColView(pfw, 2, 1, &output_peephole); // output gate
                //for GPU, the following two function can be done in one loop
                ScalePlusvElemProdv(&output_gate, 1.0f, c_fw, &output_peephole);
                Sigmoid(&output_gate, &output_gate);

                // Step 2(h). final output
                ColRowView(u, t, 1, 0, wstride, &u_fw);
                ScalePlusvElemProdv(&u_fw, 0, c_fw_nl, &output_gate);

                // keep the state for csc-blstm
                if (t == window_shift - 1) {
                    SetValue(&u_fw, buf_ufw);
                    SetValue(c_fw, buf_cfw);
                }
            }


            if (is_bidirectional) {

                SetZero(O_bw);
                PlusMatTMat(O_bw, W_bw, v);
                Plusv(O_bw, b_bw);
                FloatMatrix O_bw_col;
                FloatMatrix u_bw;
                FloatMatrix ni;

                for (int t = (int)v->num_cols - 1; t >= 0; --t) {
                    ColView(O_bw, t, 1, &O_bw_col);

                    if (t != (int)v->num_cols - 1) {
                        ColRowView(u, t + 1, 1, wstride, wstride, &u_bw);
                        PlusMatTMat(&O_bw_col, R_bw, &u_bw);
                    }

                    // peephole for input gate
                    ColRowView(O_bw, t, 1, wstride, wstride, &input_gate); // select input gate
                    if (t != (int)v->num_cols - 1) {
                        ColView(p_bw, 0, 1, &input_peephole);
                        ScalePlusvElemProdv(&input_gate, 1.0f, c_bw, &input_peephole);
                    }
                    Sigmoid(&input_gate, &input_gate);

                    ColRowView(O_bw, t, 1, wstride * 2, wstride, &forget_gate); // select forget gate
                    if (t != (int)v->num_cols - 1) {
                        ColView(p_bw, 1, 1, &forget_peephole); // forget gate
                        ScalePlusvElemProdv(&forget_gate, 1.0f, c_bw, &forget_peephole);
                    }
                    Sigmoid(&forget_gate, &forget_gate);

                    ColRowView(O_bw, t, 1, 0, wstride, &ni); // select node input
                    Tanh(&ni, &ni);

                    // Step 3(e). cell state
                    if (t != (int)v->num_cols - 1) {
                        ScalePlusvElemProdv(c_bw, 0, c_bw, &forget_gate);
                    }

                    ColRowView(O_bw, t, 1, wstride, wstride, &input_gate); // select input gate
                    ScalePlusvElemProdv(c_bw, t == (int)v->num_cols - 1 ? 0 : 1.0f, &ni, &input_gate);
                    FloorAndCeiling(c_bw, c_bw, -50, 50);


                    // Step 3(g). non-linear cell state
                    Tanh(c_bw, c_bw_nl);

                    // Step 3(f). peephole for og
                    ColRowView(O_bw, t, 1, wstride * 3, wstride, &output_gate); // select input gate
                    ColView(p_bw, 2, 1, &output_peephole); // output gate
                    ScalePlusvElemProdv(&output_gate, 1.0f, c_bw, &output_peephole);
                    Sigmoid(&output_gate, &output_gate);

                    // Step 3(h). final output
                    ColRowView(u, t, 1, wstride, wstride, &u_bw);
                    ScalePlusvElemProdv(&u_bw, 0, c_bw_nl, &output_gate);
                }
            }

        }

        // col view of the matrix
        void ColView(FloatMatrix *mat, size_t start_col, size_t num_cols, FloatMatrix *out_view) {
            start_col + num_cols <= mat->num_cols || IDEC_ERROR << "requested column out of range";
            out_view->data = mat->data + start_col*mat->col_stride;
            out_view->col_stride = mat->col_stride;
            out_view->num_rows = mat->num_rows;
            out_view->num_cols = num_cols;
        }

        // row&col view of the matrix
        void ColRowView(FloatMatrix *mat, size_t start_col, size_t num_cols, size_t start_row, size_t num_rows, FloatMatrix*out_view) {
            start_col + num_cols <= mat->num_cols || IDEC_ERROR << "requested column out of range";
            start_row + num_rows <= mat->num_rows || IDEC_ERROR << "requested row out of range";

            out_view->data = mat->data + start_col * mat->col_stride + start_row;
            out_view->num_cols = num_cols;
            out_view->num_rows = num_rows;
            out_view->col_stride = 0;
        }

#define  GetCol(x, col) (x->data+x->col_stride*col)
        void PlusMatTMat(FloatMatrix*src_dst, FloatMatrix *Mt, FloatMatrix *V)// matrix-matrix multiplication
        {
            const size_t cacheablerowsV = 512;
            const size_t cacheablecolsV = 16;
            // 512 * 16 -> 32 KB

            const size_t colstripewV = cacheablecolsV;
            const size_t rowstripehM = 128;
            const size_t dotprodstep = cacheablerowsV;
            // 128 * 512 -> 64 KB

            const size_t colstrideV = V->col_stride;

            // loop over col stripes of V
            for (size_t j0 = 0; j0 < V->num_cols; j0 += colstripewV) {
                const size_t j1 = std::min(j0 + colstripewV, V->num_cols);
                // stripe of V is columns [j0,j1)

                // loop over row stripes of M
                for (size_t i0 = 0; i0 < Mt->num_cols; i0 += rowstripehM) {
                    const size_t i1 = std::min(i0 + rowstripehM, Mt->num_cols);

                    // loop over sub-ranges of the dot product (full dot product will exceed the L1 cache)
#ifdef _MSC_VER
                    __declspec(align(16)) float patchbuffer[rowstripehM * colstripewV];    // note: don't forget column rounding
#else
                    __attribute__((aligned(16))) float patchbuffer[rowstripehM * colstripewV];    // note: don't forget column rounding
#endif
                    // 128 * 16 -> 8 KB
                    memset(patchbuffer, 0, rowstripehM * colstripewV * sizeof(float));

                    for (size_t k0 = 0; k0 < V->num_rows; k0 += dotprodstep) {
                        const size_t k1 = std::min(k0 + dotprodstep, V->num_rows);
                        //const bool first = k0 == 0;

                        for (size_t i = i0; i < i1; ++i) {
                            const size_t j14 = j1 & ~3;
                            for (size_t j = j0; j < j14; j += 4)    // grouped by 4
                            {
                                const float *row = GetCol(Mt, i) + k0;    // of length k1-k0
                                const float *cols4 = GetCol(V, j) + k0;    // of length k1-k0, stride = V.ColStride()
                                float *patchij = patchbuffer + (j - j0)*rowstripehM + (i - i0);

                                dotprod4_sse(row, cols4, colstrideV, patchij, rowstripehM, k1 - k0);
                            }
                            for (size_t j = j14; j < j1; ++j) {
                                dotprod_sse(GetCol(Mt, i) + k0, GetCol(V, j) + k0, patchbuffer + (j - j0)*rowstripehM + (i - i0), k1 - k0);
                            }
                        }
                    }

                    // assign patch
                    for (size_t j = j0; j < j1; ++j) {
                        add_sse(GetCol(src_dst, j) + i0, patchbuffer + (j - j0)*rowstripehM, i1 - i0);
                    }
                }
            }
        }

        void Plusv(FloatMatrix*src_dst, FloatMatrix*v) {

            const float *vhead = GetCol(v, 0);
            for (size_t col = 0; col < src_dst->num_cols; ++col) {
                float *u = GetCol(src_dst, col);
                plus_sse(u, vhead, src_dst->num_rows);
            }
        }
        void ScalePlusvElemProdv(FloatMatrix*src_dst, const float scale, FloatMatrix *X, FloatMatrix *Y) {
            for (size_t col = 0; col < src_dst->num_cols; ++col) {
                float *p = GetCol(src_dst, col);
                const float *pX = GetCol(X, col);
                const float *pY = GetCol(Y, col);
                scale_plus_prod_sse(p, pX, pY, src_dst->num_rows, scale);
            }
        }

        void SetZero(FloatMatrix *x) {
            for (size_t col = 0; col < x->num_cols; ++col) {
                memset(GetCol(x, col), 0, sizeof(float)*x->num_rows);
            }
        }

        void Sigmoid(FloatMatrix *in, FloatMatrix *out) {
            /*float zero = 0.0f, one = 1.0f;
            float explimit = 88.722008f;
#ifdef _MSC_VER
            __declspec(align(16)) float expbuffer;
#else
            __attribute__((aligned(16))) float expbuffer;
#endif

            for (size_t col = 0; col < in->num_cols; ++col) {
                float *src_col = GetCol(in, col);
                float *dst_col = GetCol(out, col);

                size_t row;
                for (row = 0; row < in->num_rows; row++) {
                    expbuffer = std::min<float>(-src_col[row], explimit);
                    expbuffer = expf(expbuffer);
                    dst_col[row] = one / (one + expbuffer);
                }
            }*/
            for (size_t col = 0; col < in->num_cols; ++col) {
                float *pSrcCol = GetCol(in,col);
                float *pDstCol = GetCol(out, col);

                sigmod_sse(pSrcCol, pDstCol, in->num_rows);
            }
        }

        void Tanh(FloatMatrix *in, FloatMatrix *out) {
            for (size_t col = 0; col < in->num_cols; ++col) {
                float *pSrcCol = GetCol(in, col);
                float *pDstCol = GetCol(out, col);

                tanh_sse(pSrcCol, pDstCol, in->num_rows);
            }
        }

        void FloorAndCeiling(FloatMatrix *in, FloatMatrix *out, float floor, float ceiling) {
            for (size_t i = 0; i < in->num_cols; i++) {
                float * src_col = GetCol(in, i);
                float * dst_col = GetCol(out, i);
                for (size_t j = 0; j < in->num_rows; j++) {
                    dst_col[j] = std::max(src_col[j], floor);
                    dst_col[j] = std::min(dst_col[j], ceiling);
                }
            }
        }

        void SetValue(FloatMatrix *src, FloatMatrix *dst) {
            // u = v * [1 1 1 ... 1]
#ifdef _DEBUG
            // dim check
            if (src->num_rows != dst->num_rows || src->num_cols != 1) {
                IDEC_ERROR << "dimension mismatch " << src->num_rows << " vs. " << dst->num_rows << ", " << src->num_cols << " vs. 1, " << src->col_stride << " vs. " << src->col_stride;
            }
#endif
            for (size_t col = 0; col < src->num_cols; ++col) {
                memcpy(GetCol(dst, col), GetCol(src, 0), src->num_rows*sizeof(float));
            }
        }

    };

};


#endif