#!/bin/zsh
eval "$(bin/load_dotenv .env)"
export PATH=$(pwd):$(pwd)/bin:/Applications/Houdini/Houdini19.5.569/Frameworks/Houdini.framework/Versions/Current/Resources/bin:/Applications/Houdini/Houdini19.5.569/Frameworks/Houdini.framework/Versions/Current/Resources/houdini/sbin:$PATH
# The next line enables shell command completion for yc.
if [ -f '$HOME/yandex-cloud/completion.zsh.inc' ]; then source '$HOME/yandex-cloud/completion.zsh.inc'; fi

