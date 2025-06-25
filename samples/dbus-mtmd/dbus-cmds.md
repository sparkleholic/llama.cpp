```
busctl introspect ecoai.llme /ecoai/llme --user

busctl call ecoai.llme /ecoai/llme ecoai.llme.model.getModels --userbusctl call ecoai.llme /ecoai/llme ecoai.llme.model.getModels --user
busctl call ecoai.llme /ecoai/llme ecoai.llme.model load s "qwen-vl-3b-mm" --user


busctl call ecoai.llme /ecoai/llme ecoai.llme.model getRunningModel --user

busctl call ecoai.llme /ecoai/llme ecoai.llme.model queryImage sss "qwen-vl-3b-mm-1750871766132"  "Describe this image." "/hdd/Project/llama.cpp/samples/dbus-mtmd/dogs.jpg" --user --timeout=300

```

imcompleted
```
(x) busctl call ecoai.llme /ecoai/llme ecoai.llme.manage getStatus --user
```