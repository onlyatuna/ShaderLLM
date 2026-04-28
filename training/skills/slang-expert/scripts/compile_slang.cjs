const { execSync } = require('child_process');
const path = require('path');
const fs = require('fs');

// 配置路徑
const SLANG_ROOT = path.resolve(__dirname, '../../../raw/slang-2026.7-windows-x86_64');
const SLANGC = path.join(SLANG_ROOT, 'bin/slangc.exe');
const MODULE_PATH = path.join(SLANG_ROOT, 'bin/slang-standard-module-2026.7');

function compile(source, options = {}) {
    const target = options.target || 'spirv';
    const output = options.output || source.replace('.slang', '.spv');
    const profile = options.profile || 'sm_90'; // 針對 RTX 50 系列的 TMA 特性

    let cmd = `"${SLANGC}" ${source} -target ${target} -o ${output} -profile ${profile}`;
    
    // 自動添加模組搜索路徑
    cmd += ` -I "${MODULE_PATH}"`;
    cmd += ` -I "${path.join(SLANG_ROOT, 'bin')}"`;

    console.log(`Executing: ${cmd}`);
    try {
        const result = execSync(cmd, { stdio: 'inherit' });
        console.log(`Successfully compiled ${source} -> ${output}`);
    } catch (err) {
        console.error(`Failed to compile ${source}`);
        process.exit(1);
    }
}

// 命令列介面
const args = process.argv.slice(2);
if (args.length === 0) {
    console.log("Usage: node compile_slang.cjs <source.slang> [output.spv]");
} else {
    compile(args[0], { output: args[1] });
}
