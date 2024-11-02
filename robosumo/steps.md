# Installation instructions

These instructions don't actually work so ignore them for now
## 1. Install an MPI implementation

This is for macos with homebrew, not sure how to install on windows but here's my GPT chat: https://chatgpt.com/share/67265841-8798-8006-9a53-769929cf59bb

```brew install openmpi```


## 2. Install MuJoCo
Download the MuJoCo version 2.1 binaries for Linux or OSX.
Extract the downloaded mujoco210 directory into 

```~/.mujoco/mujoco210.```


## 3. Run requirements.txt
```pip install -r requirements.txt```

## 4. Install robosumo using pip
```cd robosumo_code```

```pip install -e .```

Make sure to include the dot at the end of the pip install command.

 ```cd ..```

## 5. Getting Cython to not not work
Various commands I ran when trying to get Cython to work

```brew install gcc@9```

```export CC=clang```
```export CXX=clang++```

```brew install llvm```

```
export CC=$(brew --prefix llvm)/bin/clang
export CXX=$(brew --prefix llvm)/bin/clang++
export LDFLAGS="-L$(brew --prefix llvm)/lib"
export CPPFLAGS="-I$(brew --prefix llvm)/include"
export PATH="$(brew --prefix llvm)/bin:$PATH"

```