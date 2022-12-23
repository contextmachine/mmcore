#! /bin/bash
dir="/Applications/RhinoWIP.app/Contents/Frameworks/RhCore.framework/Versions/Current/Resources"
"$dir/dotnet/$(uname -m)/dotnet" run $dir/RhinoCode.dll script ./example.py