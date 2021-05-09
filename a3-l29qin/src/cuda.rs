// This is the skeleton for the CUDA implementation
use crate::cnn::*;
use rustacuda::function::BlockSize;
use rustacuda::launch;
use rustacuda::memory::DeviceBox;
use rustacuda::prelude::*;
use std::error::Error;
use std::ffi::CString;

// Fields need to be ordered this way so the DeviceBoxes are
// dropped before the Context. Otherwise the drop will panic.

pub struct CudaContext {
    conv_layer: DeviceBox<ConvLayer>,
    output_layer: DeviceBox<OutputLayer>,
    module: Module,
    stream: Stream,
    _context: Context,
}

impl CudaContext {
    pub fn init(cnn: &Cnn) -> Result<CudaContext, Box<dyn Error>> {
        rustacuda::init(CudaFlags::empty())?;
        let device = Device::get_device(0)?;
        let _context = Context::create_and_push(ContextFlags::MAP_HOST | ContextFlags::SCHED_AUTO, device)?;

        let conv_layer = DeviceBox::new(&cnn.conv_layer)?;
        let output_layer = DeviceBox::new(&cnn.output_layer)?;
        let ptx = CString::new(include_str!("../kernel/kernel.ptx"))?;
        let module = Module::load_from_string(&ptx)?;
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;
        Ok(
            Self {
                conv_layer,
                output_layer,
                _context,
                module,
                stream,
            }
        )
    }

    pub fn compute(&mut self, _input: &InputMatrix) -> Result<OutputVec, Box<dyn Error>> {
        let mut _conv_output = ConvOutput([[[0.0; CONV_OUT_DIM]; CONV_OUT_DIM]; CONV_LAYER_SIZE]);
        let mut _output = OutputVec([0.0; OUT_LAYER_SIZE]);
        let mut input = DeviceBox::new(_input)?;
        let mut conv_output = DeviceBox::new(&_conv_output)?;
        let mut output = DeviceBox::new(&_output)?;

        let module = &self.module;
        let stream = &self.stream;

        // convolution_layer
        unsafe {
            let result = launch!(module.convolution_layer<<<CONV_LAYER_SIZE as u32, 1, 0, stream>>>(
                input.as_device_ptr(),
                self.conv_layer.as_device_ptr(),
                conv_output.as_device_ptr()
            ));
            result?;
            stream.synchronize()?;
            conv_output.copy_to(&mut _conv_output)?;
        }

        // relu_layer
        unsafe {
            let result = launch!(module.relu_layer<<<CONV_LAYER_SIZE as u32, 1, 0, stream>>>(
                conv_output.as_device_ptr()
            ));
            result?;
            stream.synchronize()?;
            conv_output.copy_to(&mut _conv_output)?;
        }
    
        // output_layer
        unsafe {
            let result = launch!(module.output_layer<<<OUT_LAYER_SIZE as u32, 1, 0, stream>>>(
                conv_output.as_device_ptr(),
                self.output_layer.as_device_ptr(),
                output.as_device_ptr()
            ));
            result?;
            stream.synchronize()?;
            output.copy_to(&mut _output)?;
        }
        Ok(_output)
    }
}
