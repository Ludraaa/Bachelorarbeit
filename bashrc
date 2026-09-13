if [ -f /etc/bash_completion ]; then
    source /etc/bash_completion
fi

if [ -d /extern/data ]; then
    mkdir -p /extern/data/{Data,LLMdata,Models,MyModels,Configs,Results,Freebase-Setup}
fi


link_external() {
    local subdir="$1" target="$2"
    local src="/extern/data/${subdir}"
    if [ -d "$src" ]; then
        rm -rf "$target"
        mkdir -p "$(dirname "$target")"
        ln -s "$src" "$target"
    fi
}

link_external "Data"           "/workspace/data"
link_external "LLMdata"        "/workspace/LLMs/data"
link_external "Models"         "/workspace/LLMs/Models"
link_external "MyModels"       "/workspace/LLMs/MyModels"
link_external "Results"        "/workspace/results"
link_external "Freebase-Setup" "/workspace/Freebase-Setup"

if [ -d /extern/data/Configs ]; then
    cp -rn /opt/repo-configs/. /extern/data/Configs/
fi
link_external "Configs" "/workspace/configs"

source /workspace/env.sh

echo
echo 'Welcome to this Docker container, type "make help" to get some help'
echo 'If this is your first time, make sure to "cat SETUP.md" as well.'
echo
