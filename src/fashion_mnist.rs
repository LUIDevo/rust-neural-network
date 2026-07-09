use std::fs::File;
use std::io::Write;
use std::path::Path;
use std::time::Instant;

const URL: &str = "https://nnfs.io/datasets/fashion_mnist_images.zip";
const ZIP_PATH: &str = "fashion_mnist_images.zip";
const DATA_DIR: &str = "fashion_mnist_images";

// One time call to download dataset

fn download_zip() -> Result<(), Box<dyn std::error::Error>> {
    if Path::new(ZIP_PATH).exists() {
        println!("{ZIP_PATH} already present, skipping download");
        return Ok(());
    }

    let now = Instant::now();
    let content = reqwest::blocking::get(URL)?.error_for_status()?.bytes()?;

    let mut file = File::create(ZIP_PATH)?;
    file.write_all(&content)?;

    println!("Downloaded {ZIP_PATH} in {:?}", now.elapsed());
    Ok(())
}

fn extract_zip() -> Result<(), Box<dyn std::error::Error>> {
    if Path::new(DATA_DIR).exists() {
        println!("{DATA_DIR} already extracted, skipping");
        return Ok(());
    }

    let now = Instant::now();
    let file = File::open(ZIP_PATH)?;
    let mut archive = zip::ZipArchive::new(file)?;
    archive.extract(DATA_DIR)?;

    println!("Extracted to {DATA_DIR}/ in {:?}", now.elapsed());
    Ok(())
}

pub fn prepare_dataset() -> Result<(), Box<dyn std::error::Error>> {
    download_zip()?;
    extract_zip()?;
    Ok(())
}
