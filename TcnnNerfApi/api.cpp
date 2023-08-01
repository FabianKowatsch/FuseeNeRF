#include "stdafx.h"
#include "cpp_api.h"
#include "module.h"
#include "optimizer.h"
thread_local char* torch_last_err = NULL;
#define EXPORT_API extern "C" __declspec(dllexport)

#define CATCH(x) \
  try { \
    torch_last_err = 0; \
    x \
  } catch (const c10::Error e) { \
      torch_last_err = _strdup(e.what()); \
  } catch (const std::runtime_error e) { \
      torch_last_err = _strdup(e.what()); \
  }
#define CATCH_TENSOR(expr) \
    at::Tensor res = at::Tensor(); \
    CATCH(  \
        res = expr;  \
    );  \
    return ResultTensor(res);

inline torch::Tensor* ResultTensor(const at::Tensor& res)
{
    if (res.defined())
        return new torch::Tensor(res);
    else
        return NULL;
}

struct Handle2D {
    void* ptr1;
    void* ptr2;
};

EXPORT_API Handle2D forward(tcnnNerf::Module* module, torch::Tensor* input, torch::Tensor* params)
{
    auto result = module->fwd(*input, *params);
    Handle2D tuple = Handle2D();

    tcnnNerf::ContextWrapper* ctxWrapper = new tcnnNerf::ContextWrapper(std::move(std::get<0>(result).ctx));

    tuple.ptr1 = ctxWrapper;
    CATCH(torch::Tensor output = std::get<1>(result);
    tuple.ptr2 = ResultTensor(output);
    return tuple;
    );
}
EXPORT_API Handle2D backward(tcnnNerf::Module* module, tcnnNerf::ContextWrapper* ctxWrapper, torch::Tensor* input, torch::Tensor* params, torch::Tensor* output, torch::Tensor* outputGrad)
{
    auto result = module->bwd(ctxWrapper->ctx, *input, *params, *output, *outputGrad);
    Handle2D tuple = Handle2D();
    CATCH(torch::Tensor inputGrad = std::get<0>(result);
    torch::Tensor paramsGrad = std::get<1>(result);
    tuple.ptr1 = ResultTensor(inputGrad);
    tuple.ptr2 = ResultTensor(paramsGrad);
    return tuple;
    );
}
EXPORT_API torch::Tensor* density(tcnnNerf::Module* module, torch::Tensor* input, torch::Tensor* params)
{
    const torch::Tensor outputTensor = module->density(*input, *params);
    CATCH(
        torch::Tensor* output = ResultTensor(outputTensor);
        return output;
    );
}
EXPORT_API torch::Tensor* initialParams(tcnnNerf::Module* module, unsigned long seed)
{
    CATCH_TENSOR(module->initial_params(static_cast<unsigned long long>(seed)));
}
EXPORT_API unsigned int nInputDims(tcnnNerf::Module* module)
{
    return module->n_input_dims();
}
EXPORT_API unsigned int nInputDimsDensity(tcnnNerf::Module* module)
{
    return module->n_input_dims_density();
}
EXPORT_API unsigned int nParams(tcnnNerf::Module* module)
{
    return module->n_params();
}
EXPORT_API unsigned int nOutputDims(tcnnNerf::Module* module)
{
    return module->n_output_dims();
}
EXPORT_API unsigned int nOutputDimsDensity(tcnnNerf::Module* module)
{
    return module->n_output_dims_density();
}
EXPORT_API int paramPrecision(tcnnNerf::Module* module)
{
    tcnn::cpp::EPrecision p = module->param_precision();
    return static_cast<int>(p);
}
EXPORT_API int outputPrecision(tcnnNerf::Module* module)
{
    tcnn::cpp::EPrecision p = module->output_precision();
    return static_cast<int>(p);
}
EXPORT_API BSTR hyperparams(tcnnNerf::Module* module)
{
    nlohmann::json json = module->hyperparams();
    const char* str = json.dump().c_str();
    int strSize = (int)strlen(str) + 1;
    int wstrSize = MultiByteToWideChar(CP_ACP, 0, str, strSize, NULL, 0);
    OLECHAR* wstr = new OLECHAR[wstrSize];
    MultiByteToWideChar(CP_ACP, 0, str, strSize, wstr, wstrSize);
    return SysAllocString(wstr);
}
EXPORT_API BSTR name(tcnnNerf::Module* module)
{
    std::string name = module->name();
    const char* str = name.c_str();
    int strSize = (int)strlen(str) + 1;
    int wstrSize = MultiByteToWideChar(CP_ACP, 0, str, strSize, NULL, 0);
    OLECHAR* wstr = new OLECHAR[wstrSize];
    MultiByteToWideChar(CP_ACP, 0, str, strSize, wstr, wstrSize);
    return SysAllocString(wstr);
}

EXPORT_API tcnnNerf::Module* createNerfNetwork(unsigned int n_pos_dims, 
    unsigned int n_dir_dims, 
    unsigned int n_extra_dims,
    unsigned int dir_offset, 
    const char* posEncoding, 
    const char* dirEncoding,
    const char* sigmaNet,
    const char* colorNet
    )
{
    nlohmann::json posEncodingJson = nlohmann::json::parse(posEncoding);
    nlohmann::json dirEncodingJson = nlohmann::json::parse(dirEncoding);
    nlohmann::json sigmaNetJson = nlohmann::json::parse(sigmaNet);
    nlohmann::json colorNetJson = nlohmann::json::parse(colorNet);
    tcnnNerf::Module* module = new tcnnNerf::Module(new nerf::NerfModule(new ngp::NerfNetwork(n_pos_dims, n_dir_dims, n_extra_dims, dir_offset, posEncodingJson, dirEncodingJson, sigmaNetJson, colorNetJson)));
    return module;
}

EXPORT_API float* floatData(torch::Tensor* x)
{
    return x->data_ptr<float>();
}
EXPORT_API void* voidData(torch::Tensor* x)
{
    torch::Tensor t = *x;
    return tcnnNerf::void_data_ptr(t);
}
EXPORT_API void deleteModule(tcnnNerf::Module* module)
{
    delete module;
}

EXPORT_API unsigned int batchSizeGranularity()
{
    return tcnn::cpp::batch_size_granularity();
}
EXPORT_API int cudaDevice()
{
    return tcnn::cpp::cuda_device();
}
EXPORT_API void setCudaDevice(int device)
{
    tcnn::cpp::set_cuda_device(device);
}
EXPORT_API void freeTemporaryMemory()
{
    tcnn::cpp::free_temporary_memory();
}
EXPORT_API bool hasNetworks()
{
    return tcnn::cpp::has_networks();
}
EXPORT_API int preferredPrecision()
{
    int value = static_cast<int>(tcnn::cpp::preferred_precision());
    return value;
}

EXPORT_API tcnnNerf::Optimizer* createOptimizer(const char* config) {
    nlohmann::json optimizerJson = nlohmann::json::parse(config);
    return new tcnnNerf::Optimizer(optimizerJson);
}

EXPORT_API void step(tcnnNerf::Optimizer* optimizer, float lossScale, torch::Tensor* params, torch::Tensor* params_fp, torch::Tensor* gradients) {
    optimizer->step(lossScale, *params, *params_fp, *gradients);
}

EXPORT_API void allocate(tcnnNerf::Optimizer* optimizer, tcnnNerf::Module* module) {
    optimizer->allocate(module->n_params(), module->layer_sizes());
}

EXPORT_API BSTR optimizer_hyperparams(tcnnNerf::Optimizer* optimizer)
{
    nlohmann::json json = optimizer->hyperparams();
    const char* str = json.dump().c_str();
    int strSize = (int)strlen(str) + 1;
    int wstrSize = MultiByteToWideChar(CP_ACP, 0, str, strSize, NULL, 0);
    OLECHAR* wstr = new OLECHAR[wstrSize];
    MultiByteToWideChar(CP_ACP, 0, str, strSize, wstr, wstrSize);
    return SysAllocString(wstr);
}





