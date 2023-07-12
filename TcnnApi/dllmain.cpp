// dllmain.cpp : Defines the entry point for the DLL application.
#include "stdafx.h"
#include "module.h"
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

BOOL APIENTRY DllMain(HMODULE hModule,
    DWORD  ul_reason_for_call,
    LPVOID lpReserved
)
{
    switch (ul_reason_for_call)
    {
    case DLL_PROCESS_ATTACH:
    case DLL_THREAD_ATTACH:
    case DLL_THREAD_DETACH:
    case DLL_PROCESS_DETACH:
        break;
    }
    return TRUE;
}


struct Handle2D {
    void* ptr1;
    void* ptr2;
};
struct Handle3D {
    void* ptr1;
    void* ptr2;
    void* ptr3;
};
EXPORT_API Handle2D forward(tcnnModule::Module* module, torch::Tensor* input, torch::Tensor* params)
{
    auto result = module->fwd(*input, *params);
    Handle2D tuple = Handle2D();

    tcnnModule::ContextWrapper* ctxWrapper = new tcnnModule::ContextWrapper(std::move(std::get<0>(result).ctx));

    tuple.ptr1 = ctxWrapper;
    CATCH(torch::Tensor output = std::get<1>(result);
    tuple.ptr2 = ResultTensor(output);
    return tuple;
    );
}
EXPORT_API Handle2D backward(tcnnModule::Module* module, tcnnModule::ContextWrapper* ctxWrapper, torch::Tensor* input, torch::Tensor* params, torch::Tensor* output, torch::Tensor* outputGrad)
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
EXPORT_API Handle3D backwardBackwardInput(tcnnModule::Module* module, tcnnModule::ContextWrapper* ctxWrapper, torch::Tensor* input, torch::Tensor* params, torch::Tensor* inputGradDl, torch::Tensor* outputGrad)
{
    auto result = module->bwd_bwd_input(ctxWrapper->ctx, *input, *params, *inputGradDl, *outputGrad);
    Handle3D tuple = Handle3D();
    CATCH(torch::Tensor outputGradDl = std::get<0>(result);
    torch::Tensor paramsGrad = std::get<1>(result);
    torch::Tensor inputGrad = std::get<2>(result);
    tuple.ptr1 = ResultTensor(outputGradDl);
    tuple.ptr2 = ResultTensor(paramsGrad);
    tuple.ptr3 = ResultTensor(inputGrad);
    return tuple;
    );
}
EXPORT_API torch::Tensor* initialParams(tcnnModule::Module* module, unsigned long seed)
{
    CATCH_TENSOR(module->initial_params(static_cast<unsigned long long>(seed)));
}
EXPORT_API unsigned int nInputDims(tcnnModule::Module* module)
{
    return module->n_input_dims();
}
EXPORT_API unsigned int nParams(tcnnModule::Module* module)
{
    return module->n_params();
}
EXPORT_API unsigned int nOutputDims(tcnnModule::Module* module)
{
    return module->n_output_dims();
}
EXPORT_API int paramPrecision(tcnnModule::Module* module)
{
    tcnn::cpp::EPrecision p = module->param_precision();
    return static_cast<int>(p);
}
EXPORT_API int outputPrecision(tcnnModule::Module* module)
{
    tcnn::cpp::EPrecision p = module->output_precision();
    return static_cast<int>(p);
}
EXPORT_API BSTR hyperparams(tcnnModule::Module* module)
{
    nlohmann::json json = module->hyperparams();
    const char* str = json.dump().c_str();
    int strSize = strlen(str) + 1;
    int wstrSize = MultiByteToWideChar(CP_ACP, 0, str, strSize, NULL, 0);
    OLECHAR* wstr = new OLECHAR[wstrSize];
    MultiByteToWideChar(CP_ACP, 0, str, strSize, wstr, wstrSize);
    return SysAllocString(wstr);
}
EXPORT_API BSTR name(tcnnModule::Module* module)
{
    std::string name = module->name();
    const char* str = name.c_str();
    int strSize = strlen(str) + 1;
    int wstrSize = MultiByteToWideChar(CP_ACP, 0, str, strSize, NULL, 0);
    OLECHAR* wstr = new OLECHAR[wstrSize];
    MultiByteToWideChar(CP_ACP, 0, str, strSize, wstr, wstrSize);
    return SysAllocString(wstr);
}

EXPORT_API tcnnModule::Module* createNetwork(unsigned int inputDims, unsigned int outputDims, const char* network)
{
    nlohmann::json networkJson = nlohmann::json::parse(network);
    return tcnnModule::createNetwork(inputDims, outputDims, networkJson);

}
EXPORT_API tcnnModule::Module* createEncoding(unsigned int inputDims, const char* encoding, int precision)
{
    nlohmann::json encodingJson = nlohmann::json::parse(encoding);
    tcnn::cpp::EPrecision p = static_cast<tcnn::cpp::EPrecision>(precision);
    return tcnnModule::createEncoding(inputDims, encodingJson, p);

}
EXPORT_API tcnnModule::Module* createNetworkWithInputEncoding(unsigned int inputDims, unsigned int outputDims, const char* encoding, const char* network)
{
    nlohmann::json encodingJson = nlohmann::json::parse(encoding);
    nlohmann::json networkJson = nlohmann::json::parse(network);
    return tcnnModule::createNetworkWithInputEncoding(inputDims, outputDims, encodingJson, networkJson);
}

EXPORT_API float* floatData(torch::Tensor* x)
{
    return x->data_ptr<float>();
}
EXPORT_API void* voidData(torch::Tensor* x)
{
    torch::Tensor t = *x;
    return tcnnModule::void_data_ptr(t);
}
EXPORT_API void deleteModule(tcnnModule::Module* module)
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





