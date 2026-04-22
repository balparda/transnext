<!-- cspell:disable -->
<!-- auto-generated; DO NOT EDIT! see base.GenerateTyperHelpMarkdown() -->

# `gen` Command-Line Interface

```text
Usage: gen [OPTIONS] COMMAND [ARGS]...                                                                                                                    
                                                                                                                                                           
 TransNext SDXL generator and DB maker.                                                                                                                    
                                                                                                                                                           
╭─ Options ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ --version                                                                    Show version and exit.                                                     │
│ --verbose             -v                        INTEGER RANGE [0<=x<=3]      Verbosity (nothing=ERROR, -v=WARNING, -vv=INFO, -vvv=DEBUG).   │
│ --color                   --no-color                                         Force enable/disable colored output (respects NO_COLOR env var if not      │
│                                                                              provided). Defaults to having colors.                                      │
│ --host                                          TEXT                         Host for the SDNext API; default: http://127.0.0.1                         │
│                                                                                                                              │
│ --port                -p                        INTEGER RANGE [0<=x<=65535]  Port number for the SDNext API; 0 ≤ p ≤ 65535; default: 7860               │
│                                                                                                                                          │
│ --db                      --no-db                                            If True, TransNext will use/update its internal DB; False means it will    │
│                                                                              not load/use DB; default: True (DB will be used/updated)                   │
│                                                                                                                                            │
│ --sidecar                 --no-sidecar                                       If True, SDNext API will save/load a sidecar JSON file with the model      │
│                                                                              files (same directory); default: True                                      │
│                                                                                                                                       │
│ --respect-vae             --no-respect-vae                                   If True, accept override of VAE option by model; only respected if         │
│                                                                              `--sidecar` is enabled; default: True                                      │
│                                                                                                                                   │
│ --respect-pony            --no-respect-pony                                  If True, accept override of Pony option by model; only respected if        │
│                                                                              `--sidecar` is enabled; default: True                                      │
│                                                                                                                                  │
│ --respect-clip2           --no-respect-clip2                                 If True, accept override of CLIP2 option by model; only respected if       │
│                                                                              `--sidecar` is enabled; default: True                                      │
│                                                                                                                                 │
│ --out                 -o                        DIRECTORY                    The local output root directory path, ex: "~/foo/bar/"; will create        │
│                                                                              sub-directories in this directory in the format YYYY-MM-DD for the days    │
│                                                                              where images are generated; if you do not use DB (i.e., `--no-db`) this is │
│                                                                              mandatory, but with the DB activated it will store the last used output    │
│                                                                              and re-use it; default: with `--db` default is last used, with `--no-db`   │
│                                                                              no default and you must provide it                                         │
│ --install-completion                                                         Install completion for the current shell.                                  │
│ --show-completion                                                            Show completion for the current shell, to copy it or customize the         │
│                                                                              installation.                                                              │
│ --help                                                                       Show this message and exit.                                                │
╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Commands ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ markdown   Emit Markdown docs for the CLI (see README.md section "Creating a New Version").                                                             │
│ make       Query the model.                                                                                                                             │
│ reproduce  Reproduce an existing DB image by hash or file path.                                                                                         │
│ sync       Go over all known image dirs, check for new/deleted images, update DB accordingly.                                                           │
╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
                                                                                                                                                           
 Example:                                                                                                                                                  
                                                                                                                                                           
 # --- Generating Images ---                                                                                                                               
 poetry run gen -vv --out ~/foo/bar make "dark knight" -n batman --cfg 7.5 -m SDXL_model_1234 -i 30 --sampler "Euler a"                                    
                                                                                                                                                           
 # --- Reproducing an Image ---                                                                                                                            
 poetry run gen reproduce abc123def456                                                                                                                     
 poetry run gen reproduce ~/foo/bar/image.png                                                                                                              
                                                                                                                                                           
 # --- Syncing the DB ---                                                                                                                                  
 poetry run gen sync                                                                                                                                       
 poetry run gen sync ~/foo/bar/new/dir                                                                                                                     
                                                                                                                                                           
 # --- Emitting CLI Markdown Docs ---                                                                                                                      
 poetry run gen markdown > gen.md
```

## `gen make` Command

```text
Usage: gen make [OPTIONS] POSITIVE_PROMPT                                                                                                                 
                                                                                                                                                           
 Query the model.                                                                                                                                          
                                                                                                                                                           
╭─ Arguments ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ *    positive_prompt      TEXT  Query input string to guide the image generation, positive prompt; "user prompt"                              │
╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Options ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ --negative     -n                 TEXT                                                       Negative prompt to guide the image generation; "negative   │
│                                                                                              prompt"; default: no negative prompt                       │
│ --iterations   -i                 INTEGER RANGE [1<=x<=200]                                  Number of steps (iterations) for the image generation; 1 ≤ │
│                                                                                              i ≤ 200; default: 20                                       │
│                                                                                                                                            │
│ --seed         -s                 INTEGER RANGE [1<=x<=18446744073709551615]                 Seed for the image generation; 0 < s ≤                     │
│                                                                                              18446744073709551615; if not provided (default), a random  │
│                                                                                              seed will be used                                          │
│ --vseed                           INTEGER RANGE [1<=x<=18446744073709551615]                 Variation seed for the image generation; 0 < s ≤           │
│                                                                                              18446744073709551615; if not provided (default) variation  │
│                                                                                              seeds will not be used                                     │
│ --vstrength                       FLOAT RANGE [0.0<=x<=1.0]                                  Variation strength for the image generation, i.e., how     │
│                                                                                              much to mix the variation seed with the base (regular)     │
│                                                                                              seed; 0.0 ≤ s ≤ 1.0; default: 0.5; only used if variation  │
│                                                                                              seed is provided                                           │
│                                                                                                                                           │
│ --width        -w                 INTEGER RANGE [16<=x<=4096]                                Width of the generated image; 16 ≤ i ≤ 4096, multiple of   │
│                                                                                              8; default: 1024                                           │
│                                                                                                                                          │
│ --height       -h                 INTEGER RANGE [16<=x<=4096]                                Height of the generated image; 16 ≤ i ≤ 4096, multiple of  │
│                                                                                              8; default: 1024                                           │
│                                                                                                                                          │
│ --sampler                         [UniPC|DDIM|Euler|Euler a|Euler SGM|Euler EDM|Euler        Sampler to use for the generation; default: 'DPM++ SDE'    │
│                                   FlowMatch|DPM++|DPM++ 2M|DPM++ 3M|DPM++ 1S|DPM++                                                  │
│                                   SDE|DPM++ 2M SDE|DPM++ 2M EDM|DPM++ Cosine|DPM SDE|DPM++                                                              │
│                                   Inverse|DPM++ 2M Inverse|DPM++ 3M Inverse|UniPC                                                                       │
│                                   FlowMatch|DPM2 FlowMatch|DPM2a FlowMatch|DPM2++ 2M                                                                    │
│                                   FlowMatch|DPM2++ 2S FlowMatch|DPM2++ SDE FlowMatch|DPM2++                                                             │
│                                   2M SDE FlowMatch|DPM2++ 3M SDE FlowMatch|Heun|Heun                                                                    │
│                                   FlowMatch|LCM|LCM FlowMatch|DEIS|SA Solver|DC Solver|VDM                                                              │
│                                   Solver|TCD|TDD|Flash FlowMatch|PeRFlow|UFOGen|BDIA                                                                    │
│                                   DDIM|PNDM|IPNDM|DDPM|LMSD|KDPM2|KDPM2 a|CMSI|CogX                                                                     │
│                                   DDIM|DDIM Parallel|DDPM Parallel|DPM adaptive|DPM                                                                     │
│                                   fast|DPM++ 2S a|DPM++ 2S a Karras|DPM++ 2M Karras|DPM++                                                               │
│                                   3M SDE|DPM++ 3M SDE Karras]                                                                                           │
│ --parser                                                  Query parser to use for the generation; default: 'a1111'   │
│                                                                                                                                         │
│ --model        -m                 TEXT                                                       Model key to use for the generation; default: "_v10VAEFix" │
│                                                                                                                                       │
│ --clip                            INTEGER RANGE [1<=x<=12]                                   Clip skip value; 1 ≤ c ≤ 12; default: 1        │
│ --cfg          -g                 FLOAT RANGE [1.0<=x<=30.0]                                 CFG scale value (guidance scale); 1.0 ≤ c ≤ 30.0; default: │
│                                                                                              6.0                                                        │
│                                                                                                                                           │
│ --cfg-end                         FLOAT RANGE [0.0<=x<=1.0]                                  CFG scale application end (guidance end); 0.0 ≤ c ≤ 1.0;   │
│                                                                                              default: 0.8                                               │
│                                                                                                                                           │
│ --cfg-rescale                     FLOAT RANGE [0.0<=x<=1.0]                                  Adjusts the CFG guided result to reduce the tendency of    │
│                                                                                              high CFG to cause overexposure / oversaturation / burned   │
│                                                                                              highlights / harsh color shifts; you usually only want     │
│                                                                                              this for higher CFG scales `-g/--cfg` (e.g., > 7.0); 0.0 ≤ │
│                                                                                              r ≤ 1.0; default: 0.0                                      │
│                                                                                                                                           │
│ --sigma                                  Sampler sigma schedule to use for the generation; default: │
│                                                                                              None (SDNext default)                                      │
│ --spacing                                                 Sampler spacing schedule to use for the generation;        │
│                                                                                              default: None (SDNext default)                             │
│ --beta                                         Sampler beta schedule to use for the generation; default:  │
│                                                                                              None (SDNext default)                                      │
│ --prediction                            Sampler prediction type to use for the generation;         │
│                                                                                              default: None (SDNext default)                             │
│ --freeu            --no-freeu                                                                Enable/disable FreeU backbone and skip feature scaling;    │
│                                                                                              default: True                                              │
│                                                                                                                                         │
│ --b1                              FLOAT RANGE [0.0<=x<=3.0]                                  FreeU b1 backbone feature scale; 0.0 ≤ b ≤ 3.0; default:   │
│                                                                                              1.05                                                       │
│                                                                                                                                          │
│ --b2                              FLOAT RANGE [0.0<=x<=3.0]                                  FreeU b2 backbone feature scale; 0.0 ≤ b ≤ 3.0; default:   │
│                                                                                              1.1                                                        │
│                                                                                                                                           │
│ --s1                              FLOAT RANGE [0.0<=x<=3.0]                                  FreeU s1 skip feature scale; 0.0 ≤ s ≤ 3.0; reduce         │
│                                                                                              over-smoothing / unnatural detail; default: 0.75           │
│                                                                                                                                          │
│ --s2                              FLOAT RANGE [0.0<=x<=3.0]                                  FreeU s2 skip feature scale; 0.0 ≤ s ≤ 3.0; reduce         │
│                                                                                              over-smoothing / unnatural detail; default: 0.65           │
│                                                                                                                                          │
│ --backup           --no-backup                                                               If True, SDNext API server will save a backup copy of the  │
│                                                                                              generated images to its default local storage; default:    │
│                                                                                              False (images will only be saved in the TransNext DB)      │
│                                                                                                                                     │
│ --redo             --no-redo                                                                 If True, forces operation to re-do; if False (default)     │
│                                                                                              will skip unnecessary operations                           │
│                                                                                                                                       │
│ --help                                                                                       Show this message and exit.                                │
╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
                                                                                                                                                           
 Example:                                                                                                                                                  
                                                                                                                                                           
 poetry run gen -vv --out ~/foo/bar make "dark knight" -n batman --cfg 7.5 -m SDXL_model_1234 -i 30 --sampler "Euler a"
```

## `gen markdown` Command

```text
Usage: gen markdown [OPTIONS]                                                                                                                             
                                                                                                                                                           
 Emit Markdown docs for the CLI (see README.md section "Creating a New Version").                                                                          
                                                                                                                                                           
╭─ Options ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ --help          Show this message and exit.                                                                                                             │
╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
                                                                                                                                                           
 Example:                                                                                                                                                  
                                                                                                                                                           
 $ poetry run gen markdown > gen.md                                                                                                                        
 <<saves CLI doc>>
```

## `gen reproduce` Command

```text
Usage: gen reproduce [OPTIONS] HASH_OR_PATH                                                                                                               
                                                                                                                                                           
 Reproduce an existing DB image by hash or file path.                                                                                                      
                                                                                                                                                           
╭─ Arguments ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ *    hash_or_path      TEXT  Image hash (hex string) or file path to reproduce. If a path is given it will be resolved to a hash via the DB index.      │
│                                                                                                                                               │
╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Options ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ --backup    --no-backup      If True, SDNext API server will save a backup copy of the generated images to its default local storage; default: False    │
│                              (images will only be saved in the TransNext DB)                                                                            │
│                                                                                                                                     │
│ --help                       Show this message and exit.                                                                                                │
╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
                                                                                                                                                           
 Example:                                                                                                                                                  
                                                                                                                                                           
 poetry run gen reproduce abc123def456                                                                                                                     
 poetry run gen reproduce ~/foo/bar/image.png
```

## `gen sync` Command

```text
Usage: gen sync [OPTIONS] [ADD_DIR]                                                                                                                       
                                                                                                                                                           
 Go over all known image dirs, check for new/deleted images, update DB accordingly.                                                                        
                                                                                                                                                           
╭─ Arguments ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│   add_dir      [ADD_DIR]  Optional directory to add to the sync process; default: no new dir, just sync known ones.                                     │
╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Options ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ --force-api    --no-force-api      If True, SDNext API server will be required; if False (default) will still TRY to connect to API, but if not found   │
│                                    will proceed standalone                                                                                              │
│                                                                                                                                  │
│ --redo         --no-redo           If True, forces operation to re-do; if False (default) will skip unnecessary operations            │
│ --help                             Show this message and exit.                                                                                          │
╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
                                                                                                                                                           
 Example:                                                                                                                                                  
                                                                                                                                                           
 poetry run gen sync                                                                                                                                       
 poetry run gen sync ~/foo/bar/new/dir
```
