/**
 * @copyright
 * Copyright      2023  PaddlePaddle         (authors: Hui Zhang)
 *
 * @copyright
 * See LICENSE for clarification regarding multiple authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <vector>

#include "k2/csrc/paddle_util.h"

namespace k2 {

paddle::AllocationType ToPaddleDeviceType(DeviceType type) {
  switch (type) {
    case kCuda:
      return paddle::AllocationType::GPU;
    case kCpu:
      return paddle::AllocationType::CPU;
    case kUnk:  // fall-through
    default:
      K2_LOG(FATAL) << "kUnk is not supported!";
      return paddle::AllocationType::CPU;  // unreachable code
  }
}

DeviceType FromTorchDeviceType(const paddle::AllocationType &type) {
  switch (type) {
    case  paddle::AllocationType::GPU:
      return kCuda;
    case  paddle::AllocationType::CPU:
      return kCpu;
    default:
      K2_LOG(FATAL) << "Unsupported device type: " << phi::AllocationTypeStr(type)
                    << ". Only paddle::AllocationType::GPU and paddle::AllocationType::CPU are supported";
      return kUnk;  // unreachable code
  }
}

Dtype ScalarTypeToDtype(paddle::DataType scalar_type) {
  switch (scalar_type) {
    case paddle::DataType::FLOAT32:
      return kFloatDtype;
    case paddle::DataType::FLOAT64:
      return kDoubleDtype;
    case paddle::DataType::INT32:
      return kInt32Dtype;
    case paddle::DataType::INT64:
      return kInt64Dtype;
    default:
      // TODO(fangjun): add other types when needed
      K2_LOG(FATAL) << "Unsupported scalar_type: " << scalar_type;
      return kInt32Dtype;  // unreachable code
  }
}

paddle::DataType ScalarTypeFromDtype(Dtype dtype) {
  switch (dtype) {
    case kFloatDtype:
      return paddle::DataType::FLOAT32;
    case kDoubleDtype:
      return paddle::DataType::FLOAT64;
    case kInt32Dtype:
      return paddle::DataType::INT32;
    case kInt64Dtype:
      return paddle::DataType::INT64;
    default:
      // TODO(fangjun): add other types when needed
      K2_LOG(FATAL) << "Unsupported dtype: " << TraitsOf(dtype).Name();
      return paddle::DataType::UNDEFINED;  // unreachable code
  }
}

template <>
paddle::Tensor ToPaddle(Array1<Arc> &array) {
  auto device_type = ToPaddleDeviceType(array.Context()->GetDeviceType());
  int32_t device_id = array.Context()->GetDeviceId();
  auto device = paddle::Place(device_type, device_id);
  auto scalar_type = ToScalarType<int32_t>::value;
  // an Arc has 4 members
  K2_STATIC_ASSERT(sizeof(Arc) == 4 * sizeof(int32_t));
  std::vector<int64_t> sizes = {array.Dim(), 4};  // [num_rows, num_cols]
  // std::vector<int64_t> strides = {4, 1};          // in number of elements
  // auto options = torch::device(device).dtype(scalar_type);
  if (array.Dim() == 0) return paddle::empty({0, 4}, scalar_type, device);

  // NOTE: we keep a copy of `Region` inside the lambda
  // so that the returned tensor outlives the input array.
  // return torch::from_blob(
  //     array.Data(), sizes, strides,
  //     [saved_region = array.GetRegion()](void *) {}, options);

  return paddle::from_blob(
    array.Data(), sizes, scalar_type, paddle::DataLayout::NCHW, device,
    [saved_region = array.GetRegion()](void*){});
}

template <>
Array1<Arc> FromPaddle<Arc>(paddle::Tensor tensor) {
  K2_CHECK_EQ(tensor.dims().size(), 2) << "Expected dim: 2. Given: " << tensor.dims().size();
  K2_CHECK_EQ(tensor.dtype(), ToScalarType<int32_t>::value)
      << "Expected scalar type: " << ToScalarType<int32_t>::value
      << ". Given: " << tensor.dtype();

  phi::IntArray strides = phi::vectorize(phi::stride(tensor.dims()));

  K2_CHECK_EQ(strides[0], 4) << "Expected stride: 4. "
                                      << "Given: " << strides[0];

  K2_CHECK_EQ(strides[1], 1) << "Expected stride: 1. "
                                      << "Given: " << strides[1];

  K2_CHECK_EQ(tensor.numel() % 4, 0);

  auto region = NewRegion(tensor);
  Array1<Arc> ans(tensor.numel() / 4, region, 0);
  return ans;
}

Tensor FromPaddle(paddle::Tensor tensor, TensorTag) {
  Dtype dtype = ScalarTypeToDtype(tensor.dtype());
  // torch::IntArrayRef sizes = tensor.sizes();
  // torch::IntArrayRef strides = tensor.strides();
  phi::IntArray sizes = phi::vectorize(tensor.dims());
  phi::IntArray strides = phi::vectorize(phi::stride(tensor.dims()));
  Shape shape({sizes.GetData().begin(), sizes.GetData().end()}, {strides.GetData().begin(), strides.GetData().end()});

  auto region = NewRegion(tensor);
  return Tensor(dtype, shape, region, 0);
}

paddle::Tensor ToPaddle(Tensor &tensor) {
  auto device_type = ToPaddleDeviceType(tensor.Context()->GetDeviceType());
  int32_t device_id = tensor.Context()->GetDeviceId();
  auto device = paddle::Place(device_type, device_id);
  auto scalar_type = ScalarTypeFromDtype(tensor.GetDtype());
  // auto options = torch::device(device).dtype(scalar_type);

  auto dims_int32 = tensor.Dims();
  auto strides_int32 = tensor.Strides();
  std::vector<int64_t> sizes(dims_int32.begin(), dims_int32.end());
  std::vector<int64_t> strides(strides_int32.begin(), strides_int32.end());

  // NOTE: we keep a copy of `Region` inside the lambda
  // so that `torch::Tensor` always accesses valid memory.
  // This prevent the memory managed by k2::Tensor from being freed
  // as long as torch::Tensor is alive.
  // return torch::from_blob(
  //     tensor.Data(), sizes, strides,
  //     [saved_region = tensor.GetRegion()](void *) {}, options);

  return paddle::from_blob(
     tensor.Data(), sizes, scalar_type, 
     paddle::DataLayout::NCHW, device, [saved_region = tensor.GetRegion()](void*){});
}

ContextPtr GetContext(paddle::Place device) {
  if (device.GetType() == paddle::AllocationType::CPU) return GetCpuContext();

  K2_CHECK_EQ(static_cast<int8_t>(device.GetType()), static_cast<int8_t>(paddle::AllocationType::GPU));
  return GetCudaContext(device.GetDeviceId());
}

}  // namespace k2
