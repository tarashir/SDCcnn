#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/default/logging.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;

/*
    Register SDC operation
*/

REGISTER_OP("SDC")
  .Input("FS: float")
  .Input("FT: float")
  .Attr("S: float")
  .Attr("R: float")
  .Attr("d: int")
  .Output("output: float")
  .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {

    shape_inference::ShapeHandle FS_shape;
    TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 4, &FS_shape));

    shape_inference::ShapeHandle FT_shape;
    TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 4, &FT_shape));
    
    shape_inference::DimensionHandle samples = c->Dim(input_shape, 0);
    shape_inference::DimensionHandle units = c->Dim(weight_shape, 1);
    
    c->set_output(0, c->Matrix(samples, units));

    return Status::OK();
  });

/*
    SDC Operation GPU
*/
void SDCKernelLauncher(
        const float**** FS, 
        const float**** FT,
        float****** outD,
        int H,
        int W,
        int L,
        int C,
        int S,
        int R,
        int d);

class SDCOpGPU : public OpKernel {
public:
  int _SS;
  int _RR;
  int _dd;

  explicit SDCOpGPU(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context,
                   context->GetAttr("S", &_SS));
    OP_REQUIRES_OK(context,
                   context->GetAttr("R", &_RR));
    OP_REQUIRES_OK(context,
                   context->GetAttr("d", &_dd));
  }
  
  void Compute(OpKernelContext* context) override {
    //printf("SDCOpGPU\n");
    DCHECK_EQ(2, context->num_inputs());
    // get the input tensor
    const Tensor& FS = context->input(0);
    
    // get the weight tensor
    const Tensor& FT = context->input(1);
    
    // check shapes of input and weights
    const TensorShape& FS_shape = FS.shape();
    const TensorShape& FT_shape = FT.shape();
    
    //Check that inputs are 4 dimensional
    DCHECK_EQ(FS.dims(), 4);
    DCHECK_EQ(FT.dims(), 4);
    
    const int H = FS_shape.dim_size(0);
    const int W = FS_shape.dim_size(1);
    const int L = FS_shape.dim_size(2);
    const int C = FS_shape.dim_size(3);

    //Check FS and FT have same dimensions 
    DCHECK_EQ(H, FT_shape.dim_size(0));
    DCHECK_EQ(W, FT_shape.dim_size(1));
    DCHECK_EQ(L, FT_shape.dim_size(2));
    DCHECK_EQ(C, FT_shape.dim_size(3));

    // create output shape
    TensorShape output_shape;
    //printf("batch_samples: %d\n", batch_samples);
    //printf("units: %d\n", units);

    output_shape.AddDim((int)(H/_SS));
    output_shape.AddDim((int)(W/_SS));
    output_shape.AddDim((int)(L/_SS));
    output_shape.AddDim(2*_RR+1);
    output_shape.AddDim(2*_RR+1);
    output_shape.AddDim(2*_RR+1);
            
    // create output tensor
    Tensor* output = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output));
    
    // get the matricies into indexable form
    auto fs = FS.matrix<float>();
    auto ft = FT.matrix<float>();
    
    // translate tensors to arrays
    // FS and FT arrays
    float ****FSarr;//[H][W][L][C];
    float ****FTarr;//[H][W][L][C];
    FSarr = new float ***[FS_shape.dim_size(0)];
    FTarr = new float ***[FT_shape.dim_size(0)];
    for (int i = 0; i < FS_shape.dim_size(0); i++) {
      FSarr[i] = new float **[FS_shape.dim_size(1)];
      FTarr[i] = new float **[FT_shape.dim_size(1)];
      for (int j = 0; j < FS_shape.dim_size(1); j++) {
        FSarr[i][j] = new float *[FS_shape.dim_size(2)];
        FTarr[i][j] = new float *[FT_shape.dim_size(2)];
        for (int k = 0; k < FS_shape.dim_size(2); k++) {
          FSarr[i][j][k] = new float [FS_shape.dim_size(3)];
          FTarr[i][j][k] = new float [FT_shape.dim_size(3)];
          for (int l = 0; l < FT_shape.dim_size(3); l++) {
            FSarr[i][j][k][l] = fs(i,j,k,l);
            FTarr[i][j][k][l] = ft(i,j,k,l);
          }
        }
      }
    }

    // the 6D output
    float ******outD;//[(int)(H/_SS)][(int)(W/_SS)][(int)(L/_SS)][2*_RR+1][2*_RR+1][2*_RR+1];
    outD = new float *****[output_shape.dim_size(0)];
    for (int i = 0; i< output_shape.dim_size(0); i++) {
      outD[i] = new float ****[output_shape.dim_size(1)];
      for (int j= 0; j< output_shape.dim_size(1); j++) {
        outD[i][j] = new float ***[output_shape.dim_size(2)];
        for (int k= 0; k< output_shape.dim_size(2); k++) {
          outD[i][j][k] = new float **[output_shape.dim_size(3)];
          for (int l= 0; l< output_shape.dim_size(3); l++) {
            outD[i][j][k][l] = new float *[output_shape.dim_size(4)];
            for (int m= 0; m< output_shape.dim_size(4); m++) {
              outD[i][j][k][l][m] = new float [output_shape.dim_size(5)];
            }
          }
        }
      }
    }
    
    // SDCKernelLauncher
    SDCKernelLauncher(
            FSarr, 
            FTarr,
            outD,
            H,
            W, 
            L, 
            C,
            _SS,
            _RR,
            _dd
        );
    
    // transfer output to stated output
    auto output_tensor = output->matrix<float>();
    for (int i = 0; i < output_shape.dim_size(0); i++) {
      for (int j = 0; j < output_shape.dim_size(1); j++) {
        for (int k = 0; k < output_shape.dim_size(2); k++) {
          for (int l = 0; l < output_shape.dim_size(3); l++) {
            for (int m = 0; m < output_shape.dim_size(4); m++) {
              for (int n = 0; n < output_shape.dim_size(5); n++) {
                output_tensor(i,j,k,l,m,n) = outD[i][j][k][l][m][n];
              }
            }
          }
        }
      }
    }

    // free all the extra shit
    for (int i = 0; i < FS_shape.dim_size(0); i++) {
      for (int j = 0; j < FS_shape.dim_size(1); j++) {
        for (int k = 0; k < FS_shape.dim_size(2); k++) {
          delete[] FSarr[i][j][k];
          delete[] FTarr[i][j][k];
        }
        delete[] FSarr[i][j];
        delete[] FTarr[i][j];
      }
      delete[] FSarr[i];
      delete[] FTarr[i];
    }
    delete[] FSarr;
    delete[] FTarr;
    
    // delete output
    for (int i= 0; i< output_shape.dim_size(0); i++) {
      for (int j= 0; j< output_shape.dim_size(1); j++) {
        for (int k= 0; k< output_shape.dim_size(2); k++) {
          for (int l= 0; l< output_shape.dim_size(3); l++) {
            for (int m= 0; m< output_shape.dim_size(4); m++) {
              delete[] outD[i][j][k][l][m];
            }
            delete[] outD[i][j][k][l];
          }
          delete[] outD[i][j][k];
        }
        delete[] outD[i][j];
      }
      delete[] outD[i];
    }
    delete[] outD;
  }
};
REGISTER_KERNEL_BUILDER(Name("SDC").Device(DEVICE_GPU), SDCOpGPU);

/*
    SDCGrad Operation GPU
*/
REGISTER_OP("SDCGrad")
  .Input("FS: float")
  .Input("FT: float")
  .Input("dLdO: float")
  .Attr("S: float")
  .Attr("R: float")
  .Attr("d: int")
  .Output("grad_FS: float")
  .Output("grad_FT: float");

void FSGradKernelLauncher(
        const float**** FT,
        const float****** dLdO,
        float**** grad_FSarr,
        int H,
        int W,
        int L,
        int C,
        int S,
        int R,
        int d);
void FTGradKernelLauncher(
        const float**** FS,
        const float****** dLdO,
        float**** grad_FTarr,
        int H,
        int W,
        int L,
        int C,
        int S,
        int R,
        int d);

class SDCGradOpGPU : public OpKernel {
public:
  int _SS;
  int _RR;
  int _dd; 
  explicit SDCGradOpGPU(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context,
                   context->GetAttr("S", &_SS));
    OP_REQUIRES_OK(context,
                   context->GetAttr("R", &_RR));
    OP_REQUIRES_OK(context,
                   context->GetAttr("d", &_dd));
  }
  
  void Compute(OpKernelContext* context) override {
  
    //printf("SDCGradOpGPU\n");
  
    DCHECK_EQ(3, context->num_inputs());

    // get the input tensor
    const Tensor& FS = context->input(0);
    
    // get the weight tensor
    const Tensor& FT = context->input(1);

    // get the differential tensor
    const Tensor& dLdO = context->input(2);
    
    // check shapes of input and weights
    const TensorShape& FS_shape = FS.shape();
    const TensorShape& FT_shape = FT.shape();
    
    //Check that inputs are 4 dimensional
    DCHECK_EQ(FS.dims(), 4);
    DCHECK_EQ(FT.dims(), 4);
    
    const int H = FS_shape.dim_size(0);
    const int W = FS_shape.dim_size(1);
    const int L = FS_shape.dim_size(2);
    const int C = FS_shape.dim_size(3);

    //Check FS and FT have same dimensions 
    DCHECK_EQ(H, FT_shape.dim_size(0));
    DCHECK_EQ(W, FT_shape.dim_size(1));
    DCHECK_EQ(L, FT_shape.dim_size(2));
    DCHECK_EQ(C, FT_shape.dim_size(3));

    // create FS grad output shape
    TensorShape grad_FS_shape;

    grad_FS_shape.AddDim(H);
    grad_FS_shape.AddDim(W);
    grad_FS_shape.AddDim(L);
    grad_FS_shape.AddDim(C);

    // create FS grad output shape
    TensorShape grad_FT_shape;
    grad_FT_shape.AddDim(H);
    grad_FT_shape.AddDim(W);
    grad_FT_shape.AddDim(L);
    grad_FT_shape.AddDim(C);

    // create output tensor
    Tensor* grad_FS = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, grad_FS_shape, &grad_FS));
    
    Tensor* grad_FT = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(1, grad_FT_shape, &grad_FT));

    // get the matricies into indexable form
    auto fs = FS.matrix<float>();
    auto ft = FT.matrix<float>();
    

    // translate tensors to arrays
    // FS and FT arrays
    float ****FSarr;//[H][W][L][C];
    float ****FTarr;//[H][W][L][C];
    float ****grad_FSarr;//[H][W][L][C];
    float ****grad_FTarr;//[H][W][L][C];
    FSarr = new float ***[FS_shape.dim_size(0)];
    FTarr = new float ***[FT_shape.dim_size(0)];
    grad_FSarr = new float ***[grad_FS_shape.dim_size(0)];
    grad_FTarr = new float ***[grad_FT_shape.dim_size(0)];
    for (int i = 0; i < FS_shape.dim_size(0); i++) {
      FSarr[i] = new float **[FS_shape.dim_size(1)];
      FTarr[i] = new float **[FT_shape.dim_size(1)];
      grad_FSarr[i] = new float **[grad_FS_shape.dim_size(1)];
      grad_FTarr[i] = new float **[grad_FT_shape.dim_size(1)];
      for (int j = 0; j < FS_shape.dim_size(1); j++) {
        FSarr[i][j] = new float *[FS_shape.dim_size(2)];
        FTarr[i][j] = new float *[FT_shape.dim_size(2)];
        grad_FSarr[i][j] = new float *[grad_FS_shape.dim_size(2)];
        grad_FTarr[i][j] = new float *[grad_FT_shape.dim_size(2)];
        for (int k = 0; k < FS_shape.dim_size(2); k++) {
          FSarr[i][j][k] = new float [FS_shape.dim_size(3)];
          FTarr[i][j][k] = new float [FT_shape.dim_size(3)];
          grad_FSarr[i][j][k] = new float [grad_FS_shape.dim_size(3)];
          grad_FTarr[i][j][k] = new float [grad_FT_shape.dim_size(3)];
          for (int l = 0; l < FT_shape.dim_size(3); l++) {
            FSarr[i][j][k][l] = fs(i,j,k,l);
            FTarr[i][j][k][l] = ft(i,j,k,l);
          }
        }
      }
    }

    // get the values in the dldo
    auto dldo = dLdO.matrix<float>();
    const TensorShape& dLdO_shape = dLdO.shape();
    float ******dLdOarr;//[(int)(H/_SS)][(int)(W/_SS)][(int)(L/_SS)][2*_RR+1][2*_RR+1][2*_RR+1];
    dLdOarr = new float *****[dLdO_shape.dim_size(0)];
    for (int i= 0; i< dLdO_shape.dim_size(0); i++) {
      dLdOarr[i] = new float ****[dLdO_shape.dim_size(1)];
      for (int j= 0; j< dLdO_shape.dim_size(1); j++) {
        dLdOarr[i][j] = new float ***[dLdO_shape.dim_size(2)];
        for (int k= 0; k< dLdO_shape.dim_size(2); k++) {
          dLdOarr[i][j][k] = new float **[dLdO_shape.dim_size(3)];
          for (int l= 0; l< dLdO_shape.dim_size(3); l++) {
            dLdOarr[i][j][k][l] = new float *[dLdO_shape.dim_size(4)];
            for (int m= 0; m< dLdO_shape.dim_size(4); m++) {
              dLdOarr[i][j][k][l][m] = new float [dLdO_shape.dim_size(5)];
              for (int n= 0; n< dLdO_shape.dim_size(5); n++) {
                dLdOarr[i][j][k][l][m][n] = dLdOarr(i,j,k,l,m,n);
              }
            }
          }
        }
      }
    }

    // the FS and FT output arrays
    float ****grad_FS_tensor;//HWLC;
    float ****grad_FT_tensor;//HWLC;
    for (int i = 0; i < H; i++) {
      for (int j= 0; j < W; j++) {
        for (int k= 0; k < L; k++) {
          for (int c= 0; c < C; c++) {
            grad_FS[i][j][k][c];
            grad_FT[i][j][k][c];
          }
        }
      }
    }
    
    // SDCKernelLauncher
    FSGradKernelLauncher(
            FTarr,
            dLdOarr,
            grad_FSarr,
            H,
            W, 
            L, 
            C,
            _SS,
            _RR,
            _dd
        );
    FTGradKernelLauncher(
            FSarr,
            dLdOarr,
            grad_FTarr,
            H,
            W, 
            L, 
            C,
            _SS,
            _RR,
            _dd
        );
    
    // transfer result to stated output
    auto grad_FS_tensor = grad_FS->matrix<float>();
    auto grad_FT_tensor = grad_FT->matrix<float>();
    for (int i = 0; i < H; i++) {
      for (int j= 0; j < W; j++) {
        for (int k= 0; k < L; k++) {
          grad_FS_tensor(i,j,k,l) = grad_FS[i][j][k][l];
          grad_FT_tensor(i,j,k,l) = grad_FT[i][j][k][l];
        }
      }
    }

    // free all the extra shit
    for (int i = 0; i < FS_shape.dim_size(0); i++) {
      for (int j = 0; j < FS_shape.dim_size(1); j++) {
        for (int k = 0; k < FS_shape.dim_size(2); k++) {
          delete[] FSarr[i][j][k];
          delete[] FTarr[i][j][k];
          delete[] grad_FSarr[i][j][k];
          delete[] grad_FTarr[i][j][k];
        }
        delete[] FSarr[i][j];
        delete[] FTarr[i][j];
        delete[] grad_FSarr[i][j];
        delete[] grad_FTarr[i][j];
      }
      delete[] FSarr[i];
      delete[] FTarr[i];
      delete[] grad_FSarr[i];
      delete[] grad_FTarr[i];
    }
    delete[] FSarr;
    delete[] FTarr;
    delete[] grad_FSarr;
    delete[] grad_FTarr;

    for (int i= 0; i< dLdO_shape.dim_size(0); i++) {
      for (int j= 0; j< dLdO_shape.dim_size(1); j++) {
        for (int k= 0; k< dLdO_shape.dim_size(2); k++) {
          for (int l= 0; l< dLdO_shape.dim_size(3); l++) {
            for (int m= 0; m< dLdO_shape.dim_size(4); m++) {
              delete[] dLdOarr[i][j][k][l][m];
            }
            delete[] dLdOarr[i][j][k][l];
          }
          delete[] dLdOarr[i][j][k];
        }
        delete[] dLdOarr[i][j];
      }
      delete[] dLdOarr[i];
    }
    delete[] dLdOarr;

  }
};

REGISTER_KERNEL_BUILDER(Name("SDCGrad").Device(DEVICE_GPU), SDCGradOpGPU);

