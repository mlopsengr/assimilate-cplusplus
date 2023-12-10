#include "src/fastertransformer/models/bart/BartDecoder.h"

namespace fastertransformer {

template<typename T>
void BartDecoder<T>::initialize()
{
  self_attention_layer_ = 
    new TensorParallelDecoderSelfAttentionLayer<T>(max_batch_size_,
                                                    head_num_,
                                                    d_model_,
                                                    q_scaling_,
                                                    tensor_para_,
                                                    stream_,
                                                    cublas_wrapper_,
                                                    allocator_,
                                                    true,
                                                    is_free_buffer_after_forward_,
                                                    false,
                                                    0,
                                                    custom_all_reduce_comm_,
                                                    enable_custom_all_reduce_);

    // (1.0f/ sqrtf((float)size_per_head_)) 
    cross_attention_layer_ = 
      new TensorParallelDecoderCrossAttentionLayer<T>(max_batch_size_,
                                                      head_num_,
                                                      d_model_,
                                                      q_scaling_,
                                                      tensor_para_,
                                                      stream_,
                                                      cublas_wrapper_,
                                                      allocator_,
                                                      is_free_buffer_after_forward_,
                                                      custom_all_reduce_comm_,
                                                      enable_custom_all_reduce_);

  bool use_gated_activation = activation_type_ == ActivationType::GeGLU || activation_type_ == ActivationType::ReGLU
                              || activation_type_ == ActivationType::SiGLU;

  if (activation_type_ == ActivationType::Gelu || activation_type_ == ActivationType::GeGLU) {
    ffn_layer_ = new TensorParallelDecoderCrossAttentionLayer<T>(max_batch_size_,
                                                                 1,
                                                                 1,
                                                                 d_model_,
                                                                 0,
                                                                 inter_size_,
                                                                 tensor_para_,
                                                                 stream_,
                                                                 cublas_wrapper_,
                                                                 allocator_,
                                                                 true,
                                                                 is_free_buffer_after_forward_,
                                                                 false,
                                                                 0,
                                                                 use_gated_activation,
                                                                 custom_all_reduce_comm_,
                                                                 );

  }
  else if (activation_type_ == ActivationType::RElu || activation_type_ == ActivationType::ReGLU){
    ffn_layer_ = new TensorParallelReluFfnLayer<T>(max_batch_size_,
                                                   1,
                                                   1,
                                                   d_model_,
                                                   0,
                                                   inter_size_,
                                                   tensor_para_,
                                                   stream_,
                                                   cublas_wrapper_,
                                                  allocator_,
                                                  true,
                                                  is_free_buffer_after_forward_,
                                                  false,
                                                  0,
                                                  use_gated_activation,
                                                  custom_all_reduce_comm_,
                                                  enable_custom_all_reduce_); 
  }
  else if (activation_type_ == ActivationType::Silu || activation_type_ == ActivationType::SiGLU){
      ffn_layer_ = new TensorParallelSiluFfnlayer<T>(max_batch_size_,
                                                    1,
                                                    1,
                                                    d_model_,)
  }

}

}
