from typing import List, Tuple
from adalflow.utils.data import Dataset, Subset, DataLoader

from torch.utils.data import Dataset as TorchDataset

import unittest


# users need a sequence of data
class SampleData(Dataset):
    def __init__(self, data):
        self.data = data  # a sequence of data, if multiple item, use tuple or dict

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)


class SampleTorchDataset(TorchDataset):
    def __init__(self, data):
        self.data = data  # a sequence of data, if multiple item, use tuple or dict

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)


class TestSubset(unittest.TestCase):
    def test_subset(self):
        data = list(range(10))
        dataset = SampleData(data)
        subset = Subset(dataset, indices=[1, 3, 5])
        self.assertEqual(len(subset), 3)
        self.assertEqual(subset[0], 1)
        self.assertEqual(subset[1], 3)
        self.assertEqual(subset[2], 5)
        # test len
        self.assertEqual(len(subset), 3)

    def test_data_loader(self):
        data: List[Tuple] = []
        for i in range(10):
            data.append((i, i + 1))
        dataset = SampleData(data)
        loader = DataLoader(dataset, batch_size=4, shuffle=False)
        batch_0 = next(loader)
        print(f"batch_0: {batch_0}")
        self.assertEqual(len(batch_0), 2)
        self.assertIsInstance(
            batch_0, Tuple
        )  # Tuple of 2 elements, representing i, i+1 #tuple[0] will be tuple too
        self.assertEqual(batch_0[0], (0, 1, 2, 3))

        # cant subset the batch
        # subset = Subset(batch_0, indices=[0, 2])
        # print(f"subset: {subset}, len: {len(subset)}")
        # self.assertEqual(len(subset), 2)
        # self.assertEqual(subset[0], (0, 2))  # all x
        # self.assertEqual(subset[1], (1, 3))  # all y

        # for x, y in batch:
        #     print(f"x: {x}, y: {y}")
        # self.assertEqual(len(batch), 2)
        # self.assertIsInstance(batch[0], Tuple)

    def test_torch_data_loader(self):
        data: List[Tuple] = []
        for i in range(10):
            data.append({"x": i, "y": i + 1})
        dataset = SampleTorchDataset(data)
        loader = DataLoader(dataset, batch_size=2, shuffle=False)
        batch_0 = next(loader)
        print(f"batch_0: {batch_0}")
        self.assertEqual(
            len(batch_0), 2
        )  # its the best to use dict as a dataset instead of tuple
        # for i, (batch_x, batch_y) in enumerate(loader):
        #     print(f"torch batch: {x}, list(batch): {list(y)}")
        # for x, y in batch:
        #     print(f"x: {x}, y: {y}")
        # self.assertEqual(len(batch), 2)
        # self.assertIsInstance(batch[0], Tuple)


if __name__ == "__main__":
    unittest.main()
