if [ -f /etc/bash_completion ]; then
    source /etc/bash_completion
fi

source /workspace/env.sh

echo
echo 'Welcome to this Docker container, type "make help" to get some help'
echo 'If this is your first time, make sure to "cat SETUP.md" as well.'
echo