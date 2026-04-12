<!-- cspell:disable -->
<!-- auto-generated; DO NOT EDIT! see base.GenerateTyperHelpMarkdown() -->

# `gen` Command-Line Interface

```text
Usage: gen [OPTIONS] COMMAND [ARGS]...                                                                                                                    
                                                                                                                                                           
 MyCLI does amazing things!                                                                                                                                
                                                                                                                                                           
╭─ Options ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ --version                                                            Show version and exit.                                                             │
│ --verbose             -v                INTEGER RANGE [0<=x<=3]      Verbosity (nothing=ERROR, -v=WARNING, -vv=INFO, -vvv=DEBUG).           │
│ --color                   --no-color                                 Force enable/disable colored output (respects NO_COLOR env var if not provided).   │
│                                                                      Defaults to having colors.                                                         │
│ --host                                  TEXT                         Host for the SDNext API; default: http://127.0.0.1      │
│ --port                -p                INTEGER RANGE [0<=x<=65535]  Port number for the SDNext API; 0 ≤ p ≤ 65535; default: 7860        │
│ --db                      --no-db                                    If True, TransNext will use/update its internal DB; False means it will not        │
│                                                                      load/use DB; default: True (DB will be used/updated)                               │
│                                                                                                                                            │
│ --out                 -o                DIRECTORY                    The local output root directory path, ex: "~/foo/bar/"; will create                │
│                                                                      sub-directories in this directory in the format YYYY-MM-DD for the days where      │
│                                                                      images are generated; if you do not use DB (i.e., `--no-db`) this is mandatory,    │
│                                                                      but with the DB activated it will store the last used output and re-use it;        │
│                                                                      default: with `--db` default is last used, with `--no-db` no default and you must  │
│                                                                      provide it                                                                         │
│ --install-completion                                                 Install completion for the current shell.                                          │
│ --show-completion                                                    Show completion for the current shell, to copy it or customize the installation.   │
│ --help                                                               Show this message and exit.                                                        │
╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Commands ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ markdown  Emit Markdown docs for the CLI (see README.md section "Creating a New Version").                                                              │
│ make      Query the model.                                                                                                                              │
╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
```

## `gen make` Command

```text
Usage: gen make [OPTIONS] POSITIVE_PROMPT                                                                                                                 
                                                                                                                                                           
 Query the model.                                                                                                                                          
                                                                                                                                                           
╭─ Arguments ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ *    positive_prompt      TEXT  Query input string to guide the image generation, positive prompt; "user prompt"                              │
╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Options ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ --negative    -n                 TEXT                                          Negative prompt to guide the image generation; "negative prompt";        │
│                                                                                default: no negative prompt                                              │
│ --iterations  -i                 INTEGER RANGE [1<=x<=200]                     Number of steps (iterations) for the image generation; 1 ≤ i ≤ 200;      │
│                                                                                default: 20                                                              │
│                                                                                                                                            │
│ --seed        -s                 INTEGER RANGE [2<=x<=2147483647]              Seed for the image generation; 1 < s ≤ 2147483647; if not provided       │
│                                                                                (default), a random seed will be used                                    │
│ --width       -w                 INTEGER RANGE [16<=x<=2048]                   Width of the generated image; 16 ≤ i ≤ 2048; default: 512  │
│ --height      -h                 INTEGER RANGE [16<=x<=2048]                   Height of the generated image; 16 ≤ i ≤ 2048; default: 512               │
│                                                                                                                                           │
│ --sampler                        [Euler|Euler a|UniPC|DPM++ SDE|DPM++ 2M SDE]  Sampler to use for the generation; default: 'DPM++ SDE'                  │
│                                                                                                                                     │
│ --parser                                    Query parser to use for the generation; default: 'a1111'                 │
│                                                                                                                                         │
│ --model       -m                 TEXT                                          Model key to use for the generation; if not provided, the default model  │
│                                                                                will be used.                                                            │
│ --clip                           INTEGER RANGE [1<=x<=5]                       Clip skip value; 1 ≤ c ≤ 5; default: 1                       │
│ --cfg         -g                 FLOAT RANGE [1.0<=x<=30.0]                    CFG scale value (guidance scale); 1.0 ≤ c ≤ 30.0; default: 6.0           │
│                                                                                                                                           │
│ --cfg-end                        FLOAT RANGE [0.0<=x<=1.0]                     CFG scale application end (guidance end); 0.0 ≤ c ≤ 1.0; default: 0.8    │
│                                                                                                                                           │
│ --backup          --no-backup                                                  If True, SDNext API server will save a backup copy of the generated      │
│                                                                                images to its default local storage; default: False (images will only be │
│                                                                                saved in the TransNext DB)                                               │
│                                                                                                                                     │
│ --help                                                                         Show this message and exit.                                              │
╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
                                                                                                                                                           
 Example:                                                                                                                                                  
                                                                                                                                                           
 poetry run gen make "What is the capital of France?"                                                                                                      
 poetry run gen make --no-lms "Give me an onion soup recipe."
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
