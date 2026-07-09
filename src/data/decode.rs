use std::fs::File;
use png;
use std::io::BufReader;

pub fn decode_png(path: std::path::PathBuf)-> Vec<f64>{
    let file = BufReader::new(File::open(&path).expect("open png"));
    let decoder = png::Decoder::new(file);
    let mut reader = decoder.read_info().expect("read png header");
    let mut buf = vec![0u8; reader.output_buffer_size().expect("png buffer size")];
    let info = reader.next_frame(&mut buf).expect("decode png frame");
    let pixels = &buf[..info.buffer_size()];
    pixels.iter().map(|&b| (b as f64 - 127.5) / 127.5).collect::<Vec<f64>>()
}
