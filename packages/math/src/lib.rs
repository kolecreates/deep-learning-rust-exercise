pub mod linearalg {
    pub struct Tensor<T> {
        pub shape: Vec<usize>,
        pub data: Vec<T>
    }

    impl<T> Tensor<T> {
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
            self.data[self.flatten_indices(&indices)] = value;
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
            let corner_index = self.flatten_indices(corner);
            
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
