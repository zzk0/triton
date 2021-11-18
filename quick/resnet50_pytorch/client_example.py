import tritonclient.http as httpclient

if __name__ == '__main__':
    triton_client = httpclient.InferenceServerClient(url='127.0.0.1:8000')

    model_repository_index = triton_client.get_model_repository_index()
    server_meta = triton_client.get_server_metadata()
    model_meta = triton_client.get_model_metadata('resnet50_pytorch')
    model_config = triton_client.get_model_config('resnet50_pytorch')
    statistics = triton_client.get_inference_statistics()
    shm_status = triton_client.get_cuda_shared_memory_status()
    sshm_status = triton_client.get_system_shared_memory_status()
    
    server_live = triton_client.is_server_live()
    server_ready = triton_client.is_server_ready()
    model_ready = triton_client.is_model_ready('resnet50_pytorch')

    # 启动命令: ./bin/tritonserver --model-store=/models --model-control-mode explicit --load-model resnet50_pytorch
    triton_client.unload_model('resnet50_pytorch')
    triton_client.load_model('resnet50_pytorch')
    
    # shared memory
    triton_client.register_system_shared_memory('a', 'b', 1024)
    sshm_status = triton_client.get_system_shared_memory_status()
    triton_client.unregister_system_shared_memory('a')

    triton_client.close()

    with httpclient.InferenceServerClient(url='127.0.0.1:8000'):
        pass

