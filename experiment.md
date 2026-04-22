<!-- cspell:disable -->
<!-- auto-generated; DO NOT EDIT! see base.GenerateTyperHelpMarkdown() -->

# `experiment` Command-Line Interface

```text
Usage: experiment [OPTIONS] COMMAND [ARGS]...                                                                                                             
                                                                                                                                                           
 TransNext SDXL experiment manager.                                                                                                                        
                                                                                                                                                           
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
│ markdown  Emit Markdown docs for the CLI (see README.md section "Creating a New Version").                                                              │
│ new       Create and run a new experiment. An experiment varies one or more axes (CFG, sampler, model, positive/negative prompt) across a set of seed   │
│           values. Provide at least one --axis and --seeds. The order of --axis options on the command line determines the axis order in the experiment. │
╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
```

## `experiment markdown` Command

```text
Usage: experiment markdown [OPTIONS]                                                                                                                      
                                                                                                                                                           
 Emit Markdown docs for the CLI (see README.md section "Creating a New Version").                                                                          
                                                                                                                                                           
╭─ Options ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ --help          Show this message and exit.                                                                                                             │
╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
                                                                                                                                                           
 Example:                                                                                                                                                  
                                                                                                                                                           
 $ poetry run experiment markdown > experiment.md                                                                                                          
 <<saves CLI doc>>
```

## `experiment new` Command

```text
Usage: experiment new [OPTIONS] POSITIVE_PROMPT                                                                                                           
                                                                                                                                                           
 Create and run a new experiment. An experiment varies one or more axes (CFG, sampler, model, positive/negative prompt) across a set of seed values.       
 Provide at least one --axis and --seeds. The order of --axis options on the command line determines the axis order in the experiment.                     
                                                                                                                                                           
╭─ Arguments ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ *    positive_prompt      TEXT  Query input string to guide the image generation, positive prompt; "user prompt"                              │
╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Options ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│    --negative     -n                 TEXT                                                     Negative prompt to guide the image generation; "negative  │
│                                                                                               prompt"; default: no negative prompt                      │
│    --iterations   -i                 INTEGER RANGE [1<=x<=200]                                Number of steps (iterations) for the image generation; 1  │
│                                                                                               ≤ i ≤ 200; default: 20                                    │
│                                                                                                                                            │
│    --seed         -s                 INTEGER RANGE [1<=x<=18446744073709551615]               Seed for the image generation; 0 < s ≤                    │
│                                                                                               18446744073709551615; if not provided (default), a random │
│                                                                                               seed will be used                                         │
│    --vseed                           INTEGER RANGE [1<=x<=18446744073709551615]               Variation seed for the image generation; 0 < s ≤          │
│                                                                                               18446744073709551615; if not provided (default) variation │
│                                                                                               seeds will not be used                                    │
│    --vstrength                       FLOAT RANGE [0.0<=x<=1.0]                                Variation strength for the image generation, i.e., how    │
│                                                                                               much to mix the variation seed with the base (regular)    │
│                                                                                               seed; 0.0 ≤ s ≤ 1.0; default: 0.5; only used if variation │
│                                                                                               seed is provided                                          │
│                                                                                                                                           │
│    --width        -w                 INTEGER RANGE [16<=x<=4096]                              Width of the generated image; 16 ≤ i ≤ 4096, multiple of  │
│                                                                                               8; default: 1024                                          │
│                                                                                                                                          │
│    --height       -h                 INTEGER RANGE [16<=x<=4096]                              Height of the generated image; 16 ≤ i ≤ 4096, multiple of │
│                                                                                               8; default: 1024                                          │
│                                                                                                                                          │
│    --sampler                         [UniPC|DDIM|Euler|Euler a|Euler SGM|Euler EDM|Euler      Sampler to use for the generation; default: 'DPM++ SDE'   │
│                                      FlowMatch|DPM++|DPM++ 2M|DPM++ 3M|DPM++ 1S|DPM++                                               │
│                                      SDE|DPM++ 2M SDE|DPM++ 2M EDM|DPM++ Cosine|DPM                                                                     │
│                                      SDE|DPM++ Inverse|DPM++ 2M Inverse|DPM++ 3M                                                                        │
│                                      Inverse|UniPC FlowMatch|DPM2 FlowMatch|DPM2a                                                                       │
│                                      FlowMatch|DPM2++ 2M FlowMatch|DPM2++ 2S                                                                            │
│                                      FlowMatch|DPM2++ SDE FlowMatch|DPM2++ 2M SDE                                                                       │
│                                      FlowMatch|DPM2++ 3M SDE FlowMatch|Heun|Heun                                                                        │
│                                      FlowMatch|LCM|LCM FlowMatch|DEIS|SA Solver|DC                                                                      │
│                                      Solver|VDM Solver|TCD|TDD|Flash                                                                                    │
│                                      FlowMatch|PeRFlow|UFOGen|BDIA                                                                                      │
│                                      DDIM|PNDM|IPNDM|DDPM|LMSD|KDPM2|KDPM2 a|CMSI|CogX                                                                  │
│                                      DDIM|DDIM Parallel|DDPM Parallel|DPM adaptive|DPM                                                                  │
│                                      fast|DPM++ 2S a|DPM++ 2S a Karras|DPM++ 2M Karras|DPM++                                                            │
│                                      3M SDE|DPM++ 3M SDE Karras]                                                                                        │
│    --parser                                                Query parser to use for the generation; default: 'a1111'  │
│                                                                                                                                         │
│    --model        -m                 TEXT                                                     Model key to use for the generation; default:             │
│                                                                                               "_v10VAEFix"                                              │
│                                                                                                                                       │
│    --clip                            INTEGER RANGE [1<=x<=12]                                 Clip skip value; 1 ≤ c ≤ 12; default: 1       │
│    --cfg          -g                 FLOAT RANGE [1.0<=x<=30.0]                               CFG scale value (guidance scale); 1.0 ≤ c ≤ 30.0;         │
│                                                                                               default: 6.0                                              │
│                                                                                                                                           │
│    --cfg-end                         FLOAT RANGE [0.0<=x<=1.0]                                CFG scale application end (guidance end); 0.0 ≤ c ≤ 1.0;  │
│                                                                                               default: 0.8                                              │
│                                                                                                                                           │
│    --cfg-rescale                     FLOAT RANGE [0.0<=x<=1.0]                                Adjusts the CFG guided result to reduce the tendency of   │
│                                                                                               high CFG to cause overexposure / oversaturation / burned  │
│                                                                                               highlights / harsh color shifts; you usually only want    │
│                                                                                               this for higher CFG scales `-g/--cfg` (e.g., > 7.0); 0.0  │
│                                                                                               ≤ r ≤ 1.0; default: 0.0                                   │
│                                                                                                                                           │
│    --sigma                                Sampler sigma schedule to use for the generation;         │
│                                                                                               default: None (SDNext default)                            │
│    --spacing                                               Sampler spacing schedule to use for the generation;       │
│                                                                                               default: None (SDNext default)                            │
│    --beta                                       Sampler beta schedule to use for the generation; default: │
│                                                                                               None (SDNext default)                                     │
│    --prediction                          Sampler prediction type to use for the generation;        │
│                                                                                               default: None (SDNext default)                            │
│    --freeu            --no-freeu                                                              Enable/disable FreeU backbone and skip feature scaling;   │
│                                                                                               default: True                                             │
│                                                                                                                                         │
│    --b1                              FLOAT RANGE [0.0<=x<=3.0]                                FreeU b1 backbone feature scale; 0.0 ≤ b ≤ 3.0; default:  │
│                                                                                               1.05                                                      │
│                                                                                                                                          │
│    --b2                              FLOAT RANGE [0.0<=x<=3.0]                                FreeU b2 backbone feature scale; 0.0 ≤ b ≤ 3.0; default:  │
│                                                                                               1.1                                                       │
│                                                                                                                                           │
│    --s1                              FLOAT RANGE [0.0<=x<=3.0]                                FreeU s1 skip feature scale; 0.0 ≤ s ≤ 3.0; reduce        │
│                                                                                               over-smoothing / unnatural detail; default: 0.75          │
│                                                                                                                                          │
│    --s2                              FLOAT RANGE [0.0<=x<=3.0]                                FreeU s2 skip feature scale; 0.0 ≤ s ≤ 3.0; reduce        │
│                                                                                               over-smoothing / unnatural detail; default: 0.65          │
│                                                                                                                                          │
│    --backup           --no-backup                                                             If True, SDNext API server will save a backup copy of the │
│                                                                                               generated images to its default local storage; default:   │
│                                                                                               False (images will only be saved in the TransNext DB)     │
│                                                                                                                                     │
│    --redo             --no-redo                                                               If True, forces operation to re-do; if False (default)    │
│                                                                                               will skip unnecessary operations                          │
│                                                                                                                                       │
│    --seeds                           TEXT                                                     Pipe-separated list of seed values for the experiment     │
│                                                                                               runs; each seed must be 1 ≤ s ≤ 18446744073709551615 or   │
│                                                                                               0/-1 for a random seed; example: --seeds "666|-1|999";    │
│                                                                                               default: "-1" (no proper seed axis, only one random seed) │
│                                                                                                                                            │
│ *  --axis                            TEXT                                                     Experiment axis definition (repeatable, order is          │
│                                                                                               preserved); format: "KEY:VALUE1|VALUE2|..."; valid keys:  │
│                                                                                               cfg_scale (float values), sampler (names), model_hash     │
│                                                                                               (key prefixes), positive (prompt replacements), negative  │
│                                                                                               (prompt replacements); example: --axis                    │
│                                                                                               "sampler:Euler|DPM++ SDE" --axis "cfg_scale:6.0|7.5"      │
│                                                                                                                                               │
│    --help                                                                                     Show this message and exit.                               │
╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
                                                                                                                                                           
 Examples:                                                                                                                                                 
                                                                                                                                                           
 $ poetry run experiment new "a photo of a cat" --seeds "666|-1|999" --axis "sampler:Euler|DPM++ SDE"                                                      
 <<runs 2x2 grid: 2 samplers x 2 seeds>>                                                                                                                   
                                                                                                                                                           
 $ poetry run experiment new "a photo of a % animal" --seeds "42" --axis "positive:black cat|white dog" --axis "cfg_scale:6.0|9.0"                         
 <<runs 2x2x1 grid: 2 prompts x 2 CFGs x 1 seed>>
```
