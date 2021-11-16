pub mod linearalg {
    use std::ops::{Add, Div, Mul, Sub};

    pub struct Tensor<T> {
        pub shape: Vec<usize>,
        pub data: Vec<T>
    }

    pub fn tensor_exp(t: &Tensor<f32>) -> Tensor<f32> {
        let mut output = t.data.clone();
        for i in 0..t.data.len() {
            output[i] = t.data[i].exp();
        }

        Tensor { shape: t.shape.clone(), data: output }
    }

    pub fn tensor_log(t: &Tensor<f32>) -> Tensor<f32> {
        let mut output = t.data.clone();
        for i in 0..t.data.len() {
            output[i] = t.data[i].log(f32::EPSILON);
        }

        Tensor { shape: t.shape.clone(), data: output }
    }

    fn subtract<T: Sub<Output = T> + Copy>(a: &Vec<T>, b: &Vec<T>) -> Vec<T> {
        let mut size = vec![];
        for i in 0..a.len() {
            size.push(a[i] - b[i]);
        }

        size
    }

    fn get_max_slice_count(dimensions: &Vec<usize>) -> usize {
        let mut slice_count = 1;
        for i in 0..dimensions.len()-1 {
            slice_count *= dimensions[i];
        }
        slice_count
    }

    impl<T:Copy + PartialEq + Mul<Output = T> + Add<Output = T> + Div<Output = T> + PartialOrd> Tensor<T> {
        
        
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
        

        fn get_indices_of_slices(&self, start_indices: &Vec<usize>, end_indices: &Vec<usize>) -> Vec<(usize, usize)> {
            let size = subtract(end_indices, start_indices);
            let slice_count = get_max_slice_count(&size);
            let slice_size = size[size.len()-1];

            let mut output: Vec<(usize, usize)> = vec![];
            let mut slice_start_indices = start_indices.clone();
            for _slice_num in 0..slice_count {
                let slice_start = self.flatten_indices(&slice_start_indices);
                let slice_end = slice_start + slice_size;
                output.push((slice_start, slice_end));
                for j in 0..size.len()-1 {
                    let i = size.len() - j - 1;
                    let dim_size = size[i];
                    let dim_index = slice_start_indices[i];
                    if dim_index - start_indices[i] < dim_size {
                        slice_start_indices[i] += 1;
                        break;
                    }
                }
            }

            output
        }

        pub fn from_shape(shape: Vec<usize>, init_value: T) -> Tensor<T> {
            let mut size = 1;
            for i in 0..shape.len() {
                let dim_size = shape[i];
                size *= dim_size;
            }
            Tensor { shape: shape, data: vec![init_value; size] }
        }

        pub fn get_rank(&self) -> usize {
            self.shape.len()
        }

        pub fn set_element(&mut self, indices: &Vec<usize>, value: T) {
            let index = self.flatten_indices(&indices);
            self.data[index] = value;
        }

        pub fn set_elements(&mut self, start_indices: &Vec<usize>, end_indices: &Vec<usize>, values: &Tensor<T>){
            let indices_of_slices = self.get_indices_of_slices(start_indices, end_indices);
            for slice_num in 0..indices_of_slices.len() {
                let (slice_start, slice_end) = indices_of_slices[slice_num];
                let slice_size = slice_end - slice_start;
                let source_start_index = slice_num*slice_size;
                self.data[slice_start..slice_end].copy_from_slice(&values.data[source_start_index..source_start_index+slice_size]);
            }
        }

        pub fn get_elements(&self, start_indices: &Vec<usize>, end_indices: &Vec<usize>) -> Tensor<T> {
            let mut patch: Vec<T> = vec![];
            let indices_of_slices = self.get_indices_of_slices(start_indices, end_indices);
            for slice_num in 0..indices_of_slices.len() {
                let (slice_start, slice_end) = indices_of_slices[slice_num];
                let slice = &self.data[slice_start..slice_end];
                patch.extend_from_slice(slice);
            }
            

            Tensor { shape: subtract(end_indices, start_indices), data: patch }
        }

        pub fn multiply(&self, other: &Tensor<T>) -> Tensor<T> {
            let mut product: Vec<T> = vec![];

            for i in 0..self.data.len() {
                product.push(self.data[i] * other.data[i]);
            }

            Tensor { shape: self.shape.clone(), data: product }
        }

        pub fn sum(&self) -> T {
            let mut sum = self.data[0];
            for i in 1..self.data.len() {
                sum = sum + self.data[i];
            }
            sum
        }

        pub fn max(&self) -> T {
            let mut max = self.data[0];
            for i in 1..self.data.len(){
                let val = self.data[i];
                if val > max {
                    max = val;
                }
            }

            max
        }

        pub fn scalar_divide(&self, scalar: T) -> Tensor<T> {
            let mut output = self.data.clone();
            for i in 0..self.data.len() {
                output[i] = output[i] / scalar;
            }

            Tensor { shape: self.shape.clone(), data: output }
        }

        pub fn equals(&self, other: &Tensor<T>) -> bool {
            if self.shape.len() != other.shape.len() {
                return false;
            }
            for i in 0..self.shape.len() {
                if self.shape[i] != other.shape[i] {
                    return false;
                }
            }
            for i in 0..self.data.len() {
                if self.data[i] != other.data[i] {
                    return false;
                }
            }

            return true;
        }
    }
}

#[cfg(test)]
mod tests {
    
    mod linearalg {
        use crate::linearalg::Tensor;

        #[test]
        fn test_equals(){
            let t1 = Tensor::from_shape(vec![2,2], 0);
            let t2 = Tensor::from_shape(vec![4,4], 0);
            assert!(!t1.equals(&t2));
            let t3 = Tensor::from_shape(vec![4,4], 1);
            assert!(!t2.equals(&t3));
            assert!(t2.equals(&t2));
            let mut t4 = Tensor::from_shape(vec![2,2], 0);
            assert!(t4.equals(&t1));
            t4.set_element(&vec![0,1], 3);
            assert!(!t4.equals(&t1));
        }

        #[test]
        fn test_set_element(){
            let mut t  = Tensor::from_shape(vec![2,2], 0);
            t.set_element(&vec![1,1], 2);
            assert!(t.data[3] == 2);
            t  = Tensor::from_shape(vec![4,2,6], 0);
            t.set_element(&vec![3,0,5], 2);
            assert!(t.data[3*2*6+5] == 2);
        }

        #[test]
        fn test_set_elements_get_elements(){
            let mut t = Tensor::from_shape(vec![4,4], 0);
            let start = &vec![1,1];
            let end = &vec![3,3];
            let in_patch = &Tensor::from_shape(vec![2,2], 1);
            t.set_elements(start, end, in_patch);
            let out_patch = t.get_elements(start, end);

            assert!(in_patch.equals(&out_patch))
        }

        #[test]
        fn test_multiply(){
            let t1 = Tensor::from_shape(vec![2,2], 2);
            let t2 = Tensor::from_shape(vec![2,2], 2);
            let t3 = Tensor::from_shape(vec![2,2], 4);
            assert!(t2.multiply(&t1).equals(&t3));
        }

        #[test]
        fn test_sum(){
            let t1 = Tensor::from_shape(vec![2,2], 2);
            assert!(t1.sum() == 8);
        }

        #[test]
        fn test_max(){
            let t1 = Tensor { shape: vec![2,2], data: vec![1,2,3,4]};
            assert!(t1.max() == 4);
        }

        #[test]
        fn test_scalar_divide(){
            let t1 = Tensor{ shape: vec![2,2], data: vec![1.0,2.0,3.0,4.0]};
            let t2 = Tensor { shape: vec![2,2], data: vec![0.5,1.0,1.5,2.0]};
            assert!(t1.scalar_divide(2.0).equals(&t2));
        }
    }
}
