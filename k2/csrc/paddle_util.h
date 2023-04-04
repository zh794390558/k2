/**
 * @copyright
 * Copyright      2023  PaddlePaddle        (authors: Hui Zhang)
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

#ifndef K2_CSRC_PADDLE_UTIL_H_
#define K2_CSRC_PADDLE_UTIL_H_

#include <string>

#include "k2/csrc/array.h"
#include "k2/csrc/device_guard.h"
#include "k2/csrc/fsa.h"
#include "k2/csrc/log.h"
#include "k2/csrc/paddle_context.h"
#include "paddle/phi/api/include/tensor_utils.h"
#include "paddle/phi/core/ddim.h"

namespace k2 {

/* Convert k2::DeviceType to phi::AllocationType.
   Abort on failure.

   @param [in] type  We support only kCpu and kCuda at present.

   @return phi::AllocationType::GPU or phi::AllocationType::CPU.
 */
phi::AllocationType ToPaddleDeviceType(DeviceType type);

/* Convert phi::AllocationType to k2::DeviceType.
   Abort on failure.

   @param [in] type  We support only phi::AllocationType::GPU and phi::AllocationType::CPU currently.

   @return  kCpu or kCuda.
 */
DeviceType FromTorchDeviceType(const phi::AllocationType &type);

// Some versions of PyTorch do not have `c10::CppTypeToScalarType`,
// so we implement our own here.
template <typename T>
struct ToScalarType;

#define TO_SCALAR_TYPE(cpp_type, scalar_type) \
  template <>                                 \
  struct ToScalarType<cpp_type>               \
      : std::integral_constant<phi::DataType, scalar_type> {};

// TODO(fangjun): add other types if needed
TO_SCALAR_TYPE(float, phi::DataType::FLOAT32);
TO_SCALAR_TYPE(double, phi::DataType::FLOAT64);
TO_SCALAR_TYPE(int32_t, phi::DataType::INT32);
TO_SCALAR_TYPE(int64_t, phi::DataType::INT64);
TO_SCALAR_TYPE(bool, phi::DataType::BOOL);

#undef TO_SCALAR_TYPE

Dtype ScalarTypeToDtype(phi::DataType scalar_type);
phi::DataType ScalarTypeFromDtype(Dtype dtype);

/* Convert an Array1<T> to paddle::Tensor.

   @tparam T          A primitive type, e.g., int32_t, which has
                      the corresponding `ToScalarType<T>::value`.

   @param [in]  array The input array.

   @return a 1-D paddle::Tensor which shares the underlying memory
           with the input array.
 */
template <typename T>
paddle::Tensor ToPaddle(Array1<T> &array) {
  auto device_type = ToPaddleDeviceType(array.Context()->GetDeviceType());
  int32_t device_id = array.Context()->GetDeviceId();
  auto device = phi::Place(device_type, device_id);
  auto scalar_type = ToScalarType<T>::value;
  // We will call torch::from_blob below. However, if we
  // call it with an empty Array1, we'll get error:
  // RuntimeError: CUDA error: invalid argument Exception raised from
  // getDeviceFromPtr at /pytorch/aten/src/ATen/cuda/CUDADevice.h
  // Definitely we need look into this, but let's just return an empty tensor
  // when the input Array1 is empty for now.
  if (array.Dim() == 0) return paddle::empty({0}, scalar_type, device);

  // NOTE: we keep a copy of `Region` inside the lambda
  // so that `paddle::Tensor` always accesses valid memory.
  return paddle::from_blob(array.Data(),
                            {array.Dim()},
                            scalar_type,
                            phi::DataLayout::NCHW,
                            device,
                            [saved_region = array.GetRegion()](void *) {});
}

/* Convert a 1-D paddle::Tensor to an Array1<T>.

   @tparam T          A primitive type, e.g., int32_t, which has
                      the corresponding `ToScalarType<T>::value`.

   @param [in] tensor
                     The input paddle tensor.
   @return an Array1<T> sharing the underlying memory with the
           input tensor.
 */
template <typename T>
Array1<T> FromPaddle(paddle::Tensor tensor) {
  K2_CHECK_EQ(tensor.dims().size(), 1) << "Expected dim: 1. Given: " << tensor.dims().size();
  K2_CHECK_EQ(tensor.dtype(), ToScalarType<T>::value)
      << "Expected scalar type: " << ToScalarType<T>::value
      << ". Given: " << tensor.dtype();
  // Some empty tensor may have stride not equal to 1, e.g., tensor returned by
  // clone() method, it is valid here, so we won't check its strieds.
  if (tensor.numel()) {
    phi::IntArray strides = phi::vectorize(phi::stride(tensor.dims()));
    K2_CHECK_EQ(strides[0], 1)
        << "Expected stride: 1. Given: " << strides[0];
  }

  auto region = NewRegion(tensor);
  Array1<T> ans(tensor.numel(), region, 0);
  return ans;
}

/* Convert an Array1<Arc> to a paddle::Tensor.

   CAUTION: the returned tensor has dtype == paddle.int32, but
   its last column contains scores of type `float`. That is,
   the float binary pattern is reinterpreted as int.

   @param [in]  array Then input array.

   @return a 2-D paddle::Tensor, whose
           dtype == paddle.int32,
           num_rows == array.Dim(), and
           num_cols == 4
 */
template <>
paddle::Tensor ToPaddle(Array1<Arc> &array);

/* Convert a tensor to an Array1<Arc>.

  CAUTION: the given tensor's dtype is paddle.int32, but its
  last column contains scores of type `float`. That is,
  the int binary pattern is reinterpreted as float.

  @param [in]  tensor  a 2-D type with dtype == paddle.int32 and
                       num_cols == 4

  @return an Array1<Arc> sharing the underlying memory with
          the input tensor.
 */
template <>
Array1<Arc> FromPaddle<Arc>(paddle::Tensor tensor);

struct Array2Tag {};

template <typename T>
Array2<T> FromPaddle(paddle::Tensor tensor, Array2Tag) {
  K2_CHECK_EQ(tensor.dims().size(), 2) << "Expected dim: 2. Given: " << tensor.dims().size();
  K2_CHECK_EQ(tensor.dtype(), ToScalarType<T>::value)
      << "Expected scalar type: " << ToScalarType<T>::value
      << ". Given: " << tensor.dtype();

  phi::IntArray strides = phi::vectorize(phi::stride(tensor.dims()));
  K2_CHECK_EQ(strides[1], 1)
      << "Expected stride: 1. Given: " << strides[1];

  auto region = NewRegion(tensor);
  Array2<T> ans(tensor.dims()[0],     // dim0
                tensor.dims()[1],     // dim1
                strides[0],           // elem_stride0
                0,                    // byte_offset
                region);              // region
  return ans;
}

template <typename T>
paddle::Tensor ToPaddle(Array2<T> &array) {
  auto device_type = ToPaddleDeviceType(array.Context()->GetDeviceType());
  int32_t device_id = array.Context()->GetDeviceId();
  auto device = phi::Place(device_type, device_id);
  auto scalar_type = ToScalarType<T>::value;
  // auto options = torch::device(device).dtype(scalar_type);

  // If array is empty, the `array.Data()` will be a nullptr, which will
  // cause crash when calling `torch::from_blob`. Just return an empty tensor
  // here.
  if (array.Dim0() == 0 || array.Dim1() == 0)
    return paddle::empty({array.Dim0(), array.Dim1()}, scalar_type, device);

  // NOTE: we keep a copy of `Region` inside the lambda
  // so that `torch::Tensor` always accesses valid memory.
  // auto tensor = torch::from_blob(
  //     array.Data(), {array.Dim0(), array.Dim1()}, {array.ElemStride0(), 1},
  //     [saved_region = array.GetRegion()](void *) {}, options);

  auto tensor = paddle::from_blob(
    array.Data(), {array.Dim0(), array.Dim1()}, scalar_type, 
    phi::DataLayout::NCHW, device, [saved_region = array.GetRegion()](void *) {});

  return tensor;
}

struct TensorTag {};

Tensor FromPaddle(paddle::Tensor tensor, TensorTag);
paddle::Tensor ToPaddle(Tensor &tensor);

/** Create a k2 context from a paddle device.

   @param [in] device   It must be either a CPU or a GPU.

   @return Return either a CPU context or a CUDA context
           depending on the given device.
 */
ContextPtr GetContext(paddle::Place device);

inline ContextPtr GetContext(paddle::Tensor tensor) {
  return GetContext(tensor.place());
}

}  // namespace k2

#endif  // K2_CSRC_PADDLE_UTIL_H_
