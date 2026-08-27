use crate::math::rng::Rng;
use png;
use std::fs::File;
use std::io::BufReader;

pub fn decode_png(path: std::path::PathBuf) -> Vec<f32> {
    let file = BufReader::new(File::open(&path).expect("open png"));
    let decoder = png::Decoder::new(file);
    let mut reader = decoder.read_info().expect("read png header");
    let mut buf = vec![0u8; reader.output_buffer_size().expect("png buffer size")];
    let info = reader.next_frame(&mut buf).expect("decode png frame");
    let pixels = &buf[..info.buffer_size()];
    pixels
        .iter()
        .map(|&b| (b as f32 - 127.5) / 127.5)
        .collect::<Vec<f32>>()
}

pub fn shuffle_dataset(x: &mut Vec<Vec<f32>>, y: &mut Vec<usize>, rng: &mut Rng) {
    for i in (1..y.len()).rev() {
        let j = (rng.next_f32() * (i + 1) as f32) as usize;
        x.swap(i, j);
        y.swap(i, j);
    }
}
