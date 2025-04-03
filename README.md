# ECO: Enabling Efficient Context-aware Multimodal AI on Intel NPUs

## Demo Video  
[![Watch the demo](https://img.youtube.com/vi/dXbZ9i_pe9U/0.jpg)]([https://youtu.be/dXbZ9i_pe9U](https://youtu.be/9wRXog4LqbY))

## **Clone the Repository**  
First, clone this repository to your local machine:  
```bash
git clone https://github.com/arghadippurdue/ECO_Demo.git
cd ECO_Demo
```  

## **Download Required Model Files**  
Download the **pretrained** and **ckpt** model folders from this link: **[\[Google Drive Link\]](https://drive.google.com/drive/folders/1QhG7iaNmm2w5e5Z1Yks-Sntn9zZ8zV5s?usp=sharing)**  
Place them inside the base repository.  

## **Setup the Conda Environment**  
Create and activate the required conda environment:  
```bash
conda env create -f eco_demo.yaml -n eco_demo
conda activate eco_demo
```  

## **Convert PyTorch Models to OpenVINO**  
Before running the demo, convert the PyTorch models to OpenVINO format:  
```bash
mkdir ov_model
python convert_and_compare_enc_dec.py
```  
This will convert the **B3** model.  

To convert other models (e.g., **B2**), modify the following lines inside `convert_and_compare_enc_dec.py`:  

```python
backbone = 'mit_b3'
ckpt_path = "ckpt/b3_train/model-best.pth.tar"
ov_model_path = Path("ov_model/enc_dec_b3_torch_v1.xml")
```  

Change **"b3"** to **"b2"**, then run the script again:  
```bash
python convert_and_compare_enc_dec.py
```  

Now, you have the necessary **OpenVINO models** for the demo.  

---

## **Run Method**  
To execute the script, use the following command:  

```bash
python run.py <mode> <framework> --model <model> [--noise]
```  

### **Positional Arguments:**  
- **`mode`** → Specifies the data source. Choose from:  
  - `L515` → Runs in real-time using the L515 sensor  
  - `dataset` → Processes a pre-recorded dataset  

- **`framework`** → Defines the framework to be used. Options:  
  - `torch` → Runs using PyTorch  
  - `ov` → Runs using OpenVINO  

- **`--model`** (Required) → Specifies the model variant to use. Options:  
  - `b0` → Uses `mit_b0` backbone  
  - `b1` → Uses `mit_b1` backbone  
  - `b2` → Uses `mit_b2` backbone  
  - `b3` → Uses `mit_b3` backbone  

### **Optional Flags:**  
- **`--noise`** → Enables noise augmentation during execution  
- **`--depth`** → Enables depth processing (if not provided in step configuration)  
- **`--device`** → Specifies the device to use (default: `CPU`)  
- **`--experiment`** → Sets the experiment number (default: `0`)  

### **Example Usage:**  
1. Running with a dataset using PyTorch and the `b0` model:  
   ```bash
   python run.py dataset torch --model b0
   ```  
2. Running with the L515 sensor on OpenVINO, using the `b3` model with noise:  
   ```bash
   python run.py L515 ov --model b3 --noise
   ```  

---

## **Predefined Step Configurations**  
If you prefer not to manually specify each parameter, you can use the `--step` option with a value between `1` and `4`. Each step sets a predefined configuration:  

| **Step** | **Experiment** | **Mode** | **Framework** | **Model** | **Noise** | **Depth** | **Device** |
|----------|--------------|----------|---------------|-----------|-----------|-----------|------------|
| 1        | 1            | L515     | torch         | b2        | True      | True      | CPU        |
| 2        | 2            | L515     | ov            | b2        | True      | True      | NPU        |
| 3        | 3            | L515     | ov            | b2        | True      | False     | NPU        |
| 4        | 4            | L515     | ov            | b3        | True      | False     | NPU        |  

### **Usage with Step Option:**  
Instead of specifying all the parameters manually, simply run:  
```bash
python run.py --step 1
```  
This command will override other positional arguments with the configuration defined for step 1.  
