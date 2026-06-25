mod vertical;

fn create_dataset()->Vec<vertical::Sample>{
    let data = vertical::vertical_data(100, 3, 42);
    println!("wrote {} samples to src/dataset.json", data.len());
    data
}

pub struct LayerDense {
    weights: Vec<u32>,
    biases: Vec<u32>,
    dinputs: Vec<u32>,
    dweights: Vec<u32>,
    dbiases: Vec<u32>,
    dvalues: Vec<u32>,
}

fn main(){
    let data=create_dataset();
    // define forward pass
}
