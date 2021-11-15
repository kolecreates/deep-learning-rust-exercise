pub mod linearalg {
    pub struct Tensor<T:Clone> {
        pub shape: Vec<usize>,
        pub data: Vec<T>
    }

    impl<T:Clone> Tensor<T> {
        pub fn get_rank(&self) -> usize {
            self.shape.len()
        }
        fn flatten_indices(&self, indices: &Vec<usize>) -> usize {
            let mut scaled_index = 0;
            let mut scale = 1;
            for j in 0..indices.len() {
                let i = indices.len() - j - 1;
                let index = indices[i];
                scaled_index += scale * index;
                scale *= self.shape[i];
            }

            scaled_index
        }
        pub fn set_element(&mut self, indices: &Vec<usize>, value: T) {
            let index = self.flatten_indices(&indices);
            self.data[index] = value;
        }

        pub fn slice(&self, index: usize) -> &[T]{
            let mut slice_size = 1;
            for j in 0..self.shape.len()-1 {
                let i = self.shape.len() - j - 1;
                slice_size *= self.shape[i];
            }
            let slice_start = slice_size*index;
            let slice_end = slice_start + slice_size;

            &self.data[slice_start..slice_end]
        }

        pub fn get_patch(&self, corner: &Vec<usize>, size: &Vec<usize>) -> &[T] {
            let mut slice_count = 1;
            for i in 0..self.shape.len()-1 {
                let patch_dim_size = size[i];
                slice_count *= patch_dim_size;
            }

            let mut patch: Vec<T> = vec![];
            let offset = self.flatten_indices(corner);
            for i in 0..slice_count {
                let slice_start = offset;
                let slice_end = slice_start + size[size.len()-1];
                let slice = &self.data[slice_start..slice_end];
               patch.extend_from_slice(slice);
            }

            &[]
        }
    }
}

#[cfg(test)]
mod tests {
    
    mod linearalg {
        use crate::linearalg::Tensor;

        #[test]
        fn test_set_element(){
            let mut t  = Tensor { shape: vec![2,2], data: vec![0; 4] };
            t.set_element(&vec![1,1], 2);
            assert!(t.data[3] == 2);
            t  = Tensor { shape: vec![4,2,6], data: vec![0; 4*2*6] };
            t.set_element(&vec![3,0,5], 2);
            assert!(t.data[3*2*6+5] == 2);
        }
    }
}
