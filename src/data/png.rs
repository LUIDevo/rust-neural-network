use std::fs::File;
use png;

fn decode_png()-> Vec<f64>{
    let decoder = png::Decoder::new(File::open(path)?);
    let mut reader = decoder.read_info()?;
    let mut buf = vec![0u8; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buf)?;
    let pixels = &buf[..info.buffer_size()];
    pixels.iter().map(|&b| (b as f64 - 127.5) / 127.5).collect::<Vec<f64>>()
}
