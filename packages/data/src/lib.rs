use std::{fs::File, io::{BufReader, BufWriter, Read},};

use ndarray::{ArrayD, ArrayViewD,};
use png::HasParameters;

pub fn save_ndarray_as_image(t: &ArrayViewD<u8>, path: &str) {
    let f = File::create(path).expect("Failed to create image file at path");
    let ref mut buf_writer = BufWriter::new(f);
    let shape = t.shape();
    let rank = shape.len();
    let width = shape[rank-2];
    let height = shape[rank-1];
    let mut encoder = png::Encoder::new(buf_writer, width as u32, height as u32);
    
    if rank == 2 || (rank == 3 && shape[0] == 1) {
        encoder.set(png::ColorType::Grayscale);
    }else if rank == 3 && shape[0] == 3 {
        encoder.set(png::ColorType::RGB);
    }else{
        todo!();
    }
    

    encoder.set(png::BitDepth::Eight);

    let mut writer = encoder.write_header().expect("Failed to get encoder writer");

    writer.write_image_data(&t.to_slice().unwrap()).expect("Failed to write image data");
}

//follows the format specified here: http://yann.lecun.com/exdb/mnist/
pub fn parse_u8_ndarray_from_idx_file(path: &str) -> ArrayD<u8> {
    let f = File::open(path).expect("Unable to open file");
    let mut reader = BufReader::new(f);

    let mut magic_number = [0u8; 4];
    reader.read_exact(&mut magic_number).expect("Unable to read idx magic number");


    assert!(magic_number[0] == 0u8 && magic_number[1] == 0u8, "Leading zeros missing from idx magic number");

    assert!(magic_number[2] == 0x08, "unsupported idx data type");

    let num_dims = magic_number[3] as usize;

    let mut shape = vec![0usize; num_dims];

    for i in 0..num_dims {
        let mut dim_bytes = [0u8; 4];
        reader.read_exact(&mut dim_bytes).expect("Failed to read idx dimensions");
        shape[i] = msb_u8_array_to_u32(&dim_bytes) as usize;
    }

    let mut data = vec![];

    reader.read_to_end(&mut data).expect("Failed to read idx data");

    let out = ArrayD::from_shape_vec(shape.as_slice(), data).unwrap();

    out
}

//https://stackoverflow.com/a/36676814
fn msb_u8_array_to_u32(array: &[u8]) -> u32 {
    ((array[0] as u32) << 24) +
    ((array[1] as u32) << 16) +
    ((array[2] as u32) <<  8) +
    ((array[3] as u32) <<  0)
}

#[cfg(test)]
mod tests {
    use ndarray::Axis;

    use crate::{parse_u8_ndarray_from_idx_file, save_ndarray_as_image};


    #[test]
    fn test_parse_u8_from_idx_file(){
        let t1 = parse_u8_ndarray_from_idx_file("../../assets/mnist-train-images");
        let shape = t1.shape();
        assert!(shape[0] == 60000);
        assert!(shape[1] == 28);
        assert!(shape[2] == 28);

        let t2 = parse_u8_ndarray_from_idx_file("../../assets/mnist-train-labels");
        assert!(t2.shape().len() == 1 && t2.shape()[0] == 60000);
    }

    #[test]
    fn test_save_as_image(){
        let t1 = parse_u8_ndarray_from_idx_file("../../assets/mnist-train-images");
        let image = t1.index_axis(Axis(0), 0);
        let shape = image.shape();
        assert!(shape[0] == 28 && shape[1] == 28);
        save_ndarray_as_image(&image, "../../assets/test_save_tensor_as_image.png");
    }
}
