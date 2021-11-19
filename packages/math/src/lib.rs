pub mod linearalg {
    use std::{cmp, fmt::Display, ops::{Add, Div, Mul, Sub}};

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

    fn set_min<T: PartialOrd + Copy>(a: &mut Vec<T>, min: T) {
        for i in 0..a.len() {
            if a[i] < min {
                a[i] = min;
            }
        }
    }

    fn get_shape_from_range(a: &Vec<usize>, b: &Vec<usize>) -> Vec<usize> {
        let mut shape = subtract(a, b);
        set_min(&mut shape, 1);
        shape
    }

    fn get_max_slice_count(shape: &Vec<usize>) -> usize {
        let mut slice_count = 1;
        for i in 0..shape.len()-1 {
            slice_count *= shape[i];
        }
        slice_count
    }

    fn inc_indices(shape: &Vec<usize>, indices: &mut Vec<usize>) {
        let rank = shape.len();
        let mut carry = 1;
        for i in 0..rank {
            let j = rank - i - 1;
            indices[j] += carry;
            if indices[j] >= shape[j] {
                indices[j] = 0;
                carry = 1;
            }else {
                carry = 0;
            }
        }
    }

    fn flatten_indices(shape: &Vec<usize>, indices: &Vec<usize>) -> usize {
        let mut scaled_index = 0;
        let mut scale = 1;
        for j in 0..indices.len() {
            let i = indices.len() - j - 1;
            let index = indices[i];
            scaled_index += scale * index;
            scale *= shape[i];
        }

        scaled_index
    }

    impl<T:Copy + PartialEq + Mul<Output = T> + Add<Output = T> + Div<Output = T> + PartialOrd + Display + Default> Tensor<T> {
        
        
        fn flatten_indices(&self, indices: &Vec<usize>) -> usize {
            flatten_indices(&self.shape, indices)
        }

        fn flatten_indices_for_broadcast(&self, indices: &Vec<usize>) -> usize {
            let rank = self.get_rank();
            let mut adjusted = vec![0; rank];
            for i in 0..rank {
                let j = rank - i - 1;
                if self.shape[j] == 1 {
                    adjusted[j] = 0;
                }else{
                    adjusted[j] = indices[j];
                }
            }

            self.flatten_indices(&adjusted)
        }
        

        fn get_indices_of_slices(&self, start_indices: &Vec<usize>, end_indices: &Vec<usize>) -> Vec<(usize, usize)> {
            let size = get_shape_from_range(end_indices, start_indices);
            let slice_count = get_max_slice_count(&size);
            let slice_size = size[size.len()-1];
            let mut output: Vec<(usize, usize)> = vec![];
            let mut slice_start_indices = start_indices.clone();
            for _slice_num in 0..slice_count {
                let slice_start = self.flatten_indices(&slice_start_indices);
                let slice_end = slice_start + slice_size;
                output.push((slice_start, slice_end));
                for j in 0..size.len()-1 {
                    let i = size.len() - j - 2;
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
            

            Tensor { shape: get_shape_from_range(end_indices, start_indices), data: patch }
        }

        pub fn multiply(&self, b: &Tensor<T>) -> Tensor<T> {
            let out_shape = Tensor::get_broadcasted_shape(self, b);
            let indices = Tensor::get_indices_for_broadcasting(&out_shape, self, b);
            let mut out = Tensor::from_shape(out_shape, T::default());
            for i in 0..indices.len() {
                let (a_index, b_index) = indices[i];
                out.data[i] = self.data[a_index] * b.data[b_index];
            }

            out
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

        //matches the behaivor of NumPy dot function
        //docs: https://numpy.org/doc/stable/reference/generated/numpy.dot.html
        pub fn dot(&self, b: &Tensor<T>) -> Tensor<T> {
            let a = self;
            let a_dim = a.shape.len();
            let b_dim = b.shape.len();
            if a_dim == 1 && b_dim == 1 {
                assert!(a.shape[0] == b.shape[0]);
                return Tensor::from_shape(vec![1], a.multiply(&b).sum());
            }else if a_dim == 2 && b_dim == 2 {
                let rows = a.shape[0];
                let cols = b.shape[1];
                let mut product = Tensor::from_shape(vec![rows, cols], T::default());
                let a_cols = a.shape[1];
                let b_rows = b.shape[0];
                for i in 0..rows {
                    let a_row = a.get_elements(&vec![i, 0], &vec![i,a_cols]).flatten();
                    for j in 0..cols {
                        let b_col = b.get_elements(&vec![0, j], &vec![b_rows, j]).flatten();
                        product.set_element(&vec![i, j], a_row.multiply(&b_col).sum());
                    }
                }
                
                return product;
            }else if a_dim > 1 && b_dim == 1 {
                //If a is an N-D array and b is a 1-D array, it is a sum product over the last axis of a and b.
                let slice_size = a.shape[a.shape.len()-1];
                assert!(slice_size == b.shape[0]);
                let mut product = Tensor::from_shape(a.shape[0..a.shape.len()-1].to_vec(), T::default());
                let mut i = 0;
                while i < a.data.len() {
                    let slice = a.data[i..i+slice_size].to_vec();
                    let slice_tensor = Tensor { shape: vec![slice_size], data: slice.to_vec()};
                    product.data[i / slice_size] = slice_tensor.multiply(&b).sum();
                    i += slice_size;
                }

                return product;
            }

            assert!(false, "Dot not implemented for a=Nd && b=Md where M >= 2.");

            Tensor::from_shape(vec![0], T::default())
        }

        pub fn add(&self, b: &Tensor<T>) -> Tensor<T> {
            let out_shape = Tensor::get_broadcasted_shape(self, b);
            let indices = Tensor::get_indices_for_broadcasting(&out_shape, self, b);
            let mut out = Tensor::from_shape(out_shape, T::default());
            for i in 0..indices.len() {
                let (a_index, b_index) = indices[i];
                out.data[i] = self.data[a_index] + b.data[b_index];
            }

            out
        }

        //follows NumPy broadcasting rules 
        //see docs: https://numpy.org/doc/stable/user/basics.broadcasting.html
        fn get_broadcasted_shape(a: &Tensor<T>, b: &Tensor<T>) -> Vec<usize> {
            let a_rank = a.get_rank();
            let b_rank = b.get_rank();
            let min_rank = cmp::min(a_rank, b_rank);

            //check if broadcasting can be performed
            for i in 0..min_rank {
                let j = min_rank - i - 1;
                assert!(a.shape[j] == b.shape[j] || a.shape[j] == 1 || b.shape[j] == 1);
            }

            let max_rank = cmp::max(a_rank, b_rank);
            let mut out_shape = vec![0; max_rank];
            for j in 0..max_rank {
                let i = max_rank - j - 1;
                if j < min_rank {
                    out_shape[i] = cmp::max(a.shape[a.shape.len()-j-1], b.shape[b.shape.len()-j-1]);
                }else if a_rank == max_rank {
                    out_shape[i] = a.shape[i];
                }else {
                    out_shape[i] = b.shape[i];
                }
            }

            out_shape
        }

        //follows NumPy broadcasting rules 
        //see docs: https://numpy.org/doc/stable/user/basics.broadcasting.html
        fn get_indices_for_broadcasting(out_shape: &Vec<usize>, a: &Tensor<T>, b: &Tensor<T>) -> Vec<(usize,usize)> {
            let slice_count = get_max_slice_count(&out_shape);
            let element_count = slice_count * out_shape[out_shape.len()-1];
            let mut out = vec![(0,0); element_count];

            let mut indices = vec![0; out_shape.len()];
            
            for i in 0..element_count {
                let a_index = a.flatten_indices_for_broadcast(&indices);
                let b_index = b.flatten_indices_for_broadcast(&indices);
                out[i] = (a_index, b_index);
                inc_indices(&out_shape, &mut indices);
            }

            out
        }

        pub fn flatten(&self) -> Tensor<T> {
            Tensor { shape: vec![self.data.len()], data: self.data.clone() }
        }

        pub fn transpose(&self) -> Tensor<T> {
            let mut out_shape = self.shape.clone();
            out_shape.reverse();

            let mut indices = vec![0; out_shape.len()];


            let mut out_data = vec![T::default(); self.data.len()];
            for i in 0..self.data.len() {
                
                let index = flatten_indices(&self.shape, &indices);
                out_data[i] = self.data[index];
                indices.reverse();
                inc_indices(&out_shape, &mut indices);
                indices.reverse();
                
            }   
            
            Tensor { shape: out_shape, data: out_data }
        }

        pub fn print(&self){
            for i in 0..self.data.len() {
                print!("{},", self.data[i]);
            }
            println!("");
            
        }

        pub fn clone(&self) -> Tensor<T> {
            Tensor { shape: self.shape.clone(), data: self.data.clone() }
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
        use crate::linearalg::{Tensor};

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
        fn test_get_elements_with_zero_index(){
            let t1 = Tensor { shape: vec![2,2], data: vec![1,2,3,4]};
            let t2 = t1.get_elements(&vec![0,0],  &vec![0,2]);
            let t3 = Tensor { shape: vec![1,2], data: vec![1,2]};
            assert!(t2.equals(&t3));
        }

        #[test]
        fn test_get_elements_with_greater_rows() {
            let t1 = Tensor { shape: vec![3, 2], data: vec![10.0,11.0,12.0,13.0,14.0,15.0]};
            let t2 = t1.get_elements(&vec![0,0], &vec![0,2]);
            let t3 = Tensor { shape: vec![1,2], data: vec![10.0, 11.0]};
            assert!(t2.equals(&t3));
            let t4 = t1.get_elements(&vec![0,0], &vec![2,0]);
            let t5 = Tensor { shape: vec![2,1], data: vec![10.0, 12.0]};
            assert!(t4.equals(&t5));
        }

        #[test]
        fn test_multiply(){
            let t1 = Tensor::from_shape(vec![2,2], 2);
            let t2 = Tensor::from_shape(vec![2,2], 2);
            let t3 = Tensor::from_shape(vec![2,2], 4);
            assert!(t2.multiply(&t1).equals(&t3));
            let t4 = Tensor { shape: vec![5,4], data: vec![1.0; 5*4] };
            let t5 = Tensor { shape: vec![1], data: vec![2.0]};
            let t6 = Tensor { shape: vec![5,4], data: vec![2.0; 5*4] };
            assert!(t4.multiply(&t5).equals(&t6));
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

        #[test]
        fn test_dot_1d() {
            let t1 = Tensor{ shape: vec![3], data: vec![3.0,4.0,5.0]};
            let t2 = Tensor{ shape: vec![3], data: vec![7.0,8.0,9.0] };
            let product = t1.dot(&t2);
            assert!(product.data[0] == 98.0);
        }

        #[test]
        fn test_dot_2d(){
            let a = Tensor { shape: vec![2, 3], data: vec![3.0,4.0,5.0,6.0,7.0,8.0]};
            let b = Tensor { shape: vec![3, 2], data: vec![10.0,11.0,12.0,13.0,14.0,15.0]};
            let product = a.dot(&b);
            product.print();
            assert!(product.equals(&Tensor { shape: vec![2,2], data: vec![148.0, 160.0, 256.0, 277.0]}))
        }

        #[test]
        fn test_dot_2d_1d(){
            let a = Tensor { shape: vec![3,2], data: vec![1.0,2.0,3.0,4.0,5.0,6.0]};
            let b = Tensor { shape: vec![2], data: vec![7.0,8.0]};
            let c = Tensor { shape: vec![3], data: vec![23.0,53.0,83.0]};
            assert!(a.dot(&b).equals(&c));
        }

        #[test]
        fn test_dot_3d_1d(){
            let a = Tensor { shape: vec![2,2,2], data: vec![1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0]};
            let b = Tensor { shape: vec![2], data: vec![7.0,8.0]};
            let c = Tensor { shape: vec![2,2], data: vec![23.0, 21.0+32.0, 35.0+48.0, 49.0+64.0]};
            assert!(a.dot(&b).equals(&c));
        }

        #[test]
        fn test_tensor_add(){
            let t1 = Tensor { shape: vec![5,4], data: vec![1.0; 5*4] };
            let t2 = Tensor { shape: vec![1], data: vec![1.0]};
            let t3 = Tensor { shape: vec![5,4], data: vec![2.0; 5*4] };
            assert!(t1.add(&t2).equals(&t3));
            assert!(t2.add(&t1).equals(&t3));
            let t4 = Tensor { shape: vec![5,1,4], data: vec![1.0; 5*4] };
            let t5 = Tensor { shape: vec![5,4,1], data: vec![1.0; 5*4]};
            let t6 = Tensor { shape: vec![5,4,4], data: vec![2.0; 5*4*4] };
            assert!(t4.add(&t5).equals(&t6));
        }

        #[test]
        fn test_transpose(){
            let t1 = Tensor { shape: vec![3,2], data: vec![1,2,3,4,5,6]};
            let t2 = Tensor { shape: vec![2,3], data: vec![1,3,5,2,4,6]};
            assert!(t1.transpose().equals(&t2));
            assert!(t2.transpose().equals(&t1));

            let t3 = Tensor { shape: vec![3,3], data: vec![1,2,3,4,5,6,7, 8, 9]};
            let t4 = Tensor { shape: vec![3, 3], data: vec![1,4,7,2,5,8,3,6,9]};
            assert!(t3.transpose().equals(&t4));

            let t5 = Tensor { shape: vec![3,3,3], data: vec![0, 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26]};
            let t6 = Tensor { shape: vec![3,3,3], data: vec![0,9,18,3,12,21,6,15,24,1,10,19,4,13,22,7,16,25,2,11,20,5,14,23,8,17,26]};
            assert!(t5.transpose().equals(&t6));
        }
    }
}
