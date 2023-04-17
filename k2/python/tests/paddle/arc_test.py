import unittest
import numpy as np

import _k2
import paddle

class TestArc(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.devices = ['cpu']
        # if paddle.device.cuda.device_count() > 0 and k2.with_cuda:
        if paddle.device.cuda.device_count() > 0:
            cls.devices.append('gpu:0')

    def test_arc(self):
        arc = _k2.Arc(0, 1, 2, 100.1)
        self.assertEqual(arc.src_state, 0)
        self.assertEqual(arc.dest_state, 1)
        self.assertEqual(arc.label, 2)
        self.assertAlmostEqual(arc.score, 100.1, places=5)
        self.assertEqual(arc.__str__(), "0 1 2 100.1")

    def test_float_as_int(self):
        a_float = 10.3
        b_float = _k2.int_as_float(_k2.float_as_int(a_float))
        self.assertAlmostEqual(a_float, b_float, places=5)

    def test_int_as_float(self):
        a_int = 10
        b_int = _k2.float_as_int(_k2.int_as_float(a_int))
        self.assertEqual(a_int, b_int)
       
    def test_as_int(self):
        for device in self.devices:
            paddle.device.set_device(device)
            a_float = paddle.ones([5]) * 5.4
            b_float = _k2.as_float(_k2.as_int(a_float))
            self.assertEqual(b_float.place.__str__(), a_float.place.__str__())
            np.testing.assert_allclose(b_float.numpy(), a_float.numpy(), atol=1e-6)

    def test_as_int_empty(self):
        for device in self.devices:
            paddle.device.set_device(device)
            a_float = paddle.ones([], paddle.float32)
            b_float = _k2.as_float(_k2.as_int(a_float))
            self.assertListEqual(a_float.shape, b_float.shape)
            self.assertEqual(len(b_float.shape), 0)
            self.assertEqual(b_float.place.__str__(), a_float.place.__str__())

    def test_as_int_int(self):
        for device in self.devices:
            paddle.device.set_device(device)
            a_int = paddle.ones([10], paddle.int32)
            b_int = _k2.as_int(a_int)
            np.testing.assert_array_equal(a_int.numpy(), b_int.numpy())
            self.assertEqual(a_int.place.__str__(), b_int.place.__str__())

    def test_as_float(self):
        for device in self.devices:
            paddle.device.set_device(device)
            a_int = paddle.ones([3], paddle.int32) * 5
            b_int = _k2.as_int(_k2.as_float(a_int))
            np.testing.assert_array_equal(b_int.numpy(),  a_int.numpy())
            self.assertEqual(a_int.place.__str__(), b_int.place.__str__())

    def test_as_float_empty(self):
        for device in self.devices:
            paddle.device.set_device(device)
            a_int = paddle.ones([], paddle.int32)
            b_int = _k2.as_int(_k2.as_float(a_int))
            self.assertListEqual(a_int.shape, b_int.shape)
            self.assertEqual(len(b_int.shape), 0)
            self.assertEqual(a_int.place.__str__(), b_int.place.__str__())

if __name__ == '__main__':
    unittest.main()
