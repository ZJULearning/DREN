#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/softmax_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void SoftmaxLossForwardGPU(const int nthreads,
          const Dtype* prob_data, const Dtype* label, Dtype* loss,
          const int num, const int dim, const int spatial_dim,
          const bool has_ignore_label_, const int ignore_label_,
          Dtype* counts) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int n = index / spatial_dim;
    const int s = index % spatial_dim;
    const int label_value = static_cast<int>(label[n * spatial_dim + s]);
    if (has_ignore_label_ && label_value == ignore_label_) {
      loss[index] = 0;
      counts[index] = 0;
    } else {
      loss[index] = -log(max(prob_data[n * dim + label_value * spatial_dim + s],
                      Dtype(FLT_MIN)));
      counts[index] = 1;
    }
  }
}

template <typename Dtype>
void SoftmaxWithLossLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  if(cross_entropy_){
	  // The forward pass computes the softmax prob values.
	  softmax_layer_->Forward(softmax_bottom_vec_, softmax_top_vec_);
	  const Dtype* prob_data = prob_.cpu_data();
	  const Dtype* label = bottom[1]->cpu_data();
	  int dim = prob_.count() / outer_num_;
	  int count = 0;
	  Dtype loss = 0;
	  for (int i = 0; i < outer_num_; ++i) {
		  
		for (int j = 0; j < inner_num_; j++) {
		  const int label_value = static_cast<int>(label[i * inner_num_ + j]);
		  if (has_ignore_label_ && label_value == ignore_label_) {
			continue;
		  }
		  DCHECK_GE(label_value, 0);
		  DCHECK_LT(label_value, prob_.shape(softmax_axis_));
		 /********************************************************************/
		 //add cross_entropy mode 
		 //author: Junying Li data:2017/1/3
		 /********************************************************************/ 
		  if(cross_entropy_){
			  for (int k = 0; k < prob_.shape(softmax_axis_); k++) {
				  loss -= (k == label_value)?log(std::max(prob_data[i * dim + k * inner_num_ + j],Dtype(FLT_MIN))):
					log(std::max(1-prob_data[i * dim + k * inner_num_ + j],Dtype(FLT_MIN)));
				}
		  }
		  else{
			  loss -= log(std::max(prob_data[i * dim + label_value * inner_num_ + j],
								   Dtype(FLT_MIN)));
		  }
		  ++count;
		  /********************************************************************/ 
		}
		
	  }
	  top[0]->mutable_cpu_data()[0] = loss / get_normalizer(normalization_, count);
	  if (top.size() == 2) {
		top[1]->ShareData(prob_);
	  }
  }
  else{
	  softmax_layer_->Forward(softmax_bottom_vec_, softmax_top_vec_);
	  const Dtype* prob_data = prob_.gpu_data();
	  const Dtype* label = bottom[1]->gpu_data();
	  const int dim = prob_.count() / outer_num_;
	  const int nthreads = outer_num_ * inner_num_;
	  // Since this memory is not used for anything until it is overwritten
	  // on the backward pass, we use it here to avoid having to allocate new GPU
	  // memory to accumulate intermediate results in the kernel.
	  Dtype* loss_data = bottom[0]->mutable_gpu_diff();
	  // Similarly, this memory is never used elsewhere, and thus we can use it
	  // to avoid having to allocate additional GPU memory.
	  Dtype* counts = prob_.mutable_gpu_diff();
	  // NOLINT_NEXT_LINE(whitespace/operators)
	  SoftmaxLossForwardGPU<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
		  CAFFE_CUDA_NUM_THREADS>>>(nthreads, prob_data, label, loss_data,
		  outer_num_, dim, inner_num_, has_ignore_label_, ignore_label_, counts);
	  Dtype loss;
	  caffe_gpu_asum(nthreads, loss_data, &loss);
	  Dtype valid_count = -1;
	  // Only launch another CUDA kernel if we actually need the count of valid
	  // outputs.
	  if (normalization_ == LossParameter_NormalizationMode_VALID &&
		  has_ignore_label_) {
		caffe_gpu_asum(nthreads, counts, &valid_count);
	  }
	  top[0]->mutable_cpu_data()[0] = loss / get_normalizer(normalization_,
															valid_count);
	  if (top.size() == 2) {
		top[1]->ShareData(prob_);
	  }
	}
}

template <typename Dtype>
__global__ void SoftmaxLossBackwardGPU(const int nthreads, const Dtype* top,
          const Dtype* label, Dtype* bottom_diff, const int num, const int dim,
          const int spatial_dim, const bool has_ignore_label_,
          const int ignore_label_, Dtype* counts) {

  const int channels = dim / spatial_dim;

  CUDA_KERNEL_LOOP(index, nthreads) {
    const int n = index / spatial_dim;
    const int s = index % spatial_dim;
    const int label_value = static_cast<int>(label[n * spatial_dim + s]);

    if (has_ignore_label_ && label_value == ignore_label_) {
      for (int c = 0; c < channels; ++c) {
        bottom_diff[n * dim + c * spatial_dim + s] = 0;
      }
      counts[index] = 0;
    } else {
      bottom_diff[n * dim + label_value * spatial_dim + s] -= 1;
      counts[index] = 1;
    }
  }
}

template <typename Dtype>
void SoftmaxWithLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if(cross_entropy_){
	  if (propagate_down[1]) {
		LOG(FATAL) << this->type()
				   << " Layer cannot backpropagate to label inputs.";
	  }
	  if (propagate_down[0]) {
		Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
		const Dtype* prob_data = prob_.cpu_data();
		caffe_copy(prob_.count(), prob_data, bottom_diff);
		const Dtype* label = bottom[1]->cpu_data();
		int dim = prob_.count() / outer_num_;
		int count = 0;
		for (int i = 0; i < outer_num_; ++i) {
		  for (int j = 0; j < inner_num_; ++j) {
			const int label_value = static_cast<int>(label[i * inner_num_ + j]);
			if (has_ignore_label_ && label_value == ignore_label_) {
			  for (int c = 0; c < bottom[0]->shape(softmax_axis_); ++c) {
				bottom_diff[i * dim + c * inner_num_ + j] = 0;
			  }
			} else {
				/********************************************************************/
				//add cross_entropy mode 
				//author: Junying Li data:2017/1/3
				/********************************************************************/
			  if(cross_entropy_){
				  bottom_diff[i * dim + label_value * inner_num_ + j] -= 1;
			  }
			  else{
				  bottom_diff[i * dim + label_value * inner_num_ + j] -= 1;
			  }
			  ++count;
			  /********************************************************************/
			}
		  }
		}
		// Scale gradient
		Dtype loss_weight = top[0]->cpu_diff()[0] /
							get_normalizer(normalization_, count);
		caffe_scal(prob_.count(), loss_weight, bottom_diff);
	  }
  }
  else{
	  
	  if (propagate_down[1]) {
		LOG(FATAL) << this->type()
				   << " Layer cannot backpropagate to label inputs.";
	  }
	  if (propagate_down[0]) {
		Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
		const Dtype* prob_data = prob_.gpu_data();
		const Dtype* top_data = top[0]->gpu_data();
		caffe_gpu_memcpy(prob_.count() * sizeof(Dtype), prob_data, bottom_diff);
		const Dtype* label = bottom[1]->gpu_data();
		const int dim = prob_.count() / outer_num_;
		const int nthreads = outer_num_ * inner_num_;
		// Since this memory is never used for anything else,
		// we use to to avoid allocating new GPU memory.
		Dtype* counts = prob_.mutable_gpu_diff();
		// NOLINT_NEXT_LINE(whitespace/operators)
		SoftmaxLossBackwardGPU<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
			CAFFE_CUDA_NUM_THREADS>>>(nthreads, top_data, label, bottom_diff,
			outer_num_, dim, inner_num_, has_ignore_label_, ignore_label_, counts);

		Dtype valid_count = -1;
		// Only launch another CUDA kernel if we actually need the count of valid
		// outputs.
		if (normalization_ == LossParameter_NormalizationMode_VALID &&
			has_ignore_label_) {
		  caffe_gpu_asum(nthreads, counts, &valid_count);
		}
		const Dtype loss_weight = top[0]->cpu_diff()[0] /
								  get_normalizer(normalization_, valid_count);
		caffe_gpu_scal(prob_.count(), loss_weight , bottom_diff);
	  }
	}
}

INSTANTIATE_LAYER_GPU_FUNCS(SoftmaxWithLossLayer);

}  // namespace caffe
