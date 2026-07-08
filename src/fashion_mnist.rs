use reqwest::get;
use std::{fs::File, time::Instant};

fn download_training_data() -> Result<(), Box<dyn std::error::Error>> {
    let URL = "https://nnfs.io/datasets/fashion_mnist_images.zip";
    let FILE_NAME = "fashion_mnist";
    let now = Instant::now();
    let response = get()?;
    let content = response.bytes()?;

    let mut downloaded_file = File::create(FILE_NAME)?;
    downloaded_file.write_all(&content)?;

    let duration = now.elapsed();
    println!("Downloaded file in {duration:?}");
    Ok(())
}
