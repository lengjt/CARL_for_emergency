// Execute standalone replay scripts with a minimal DOM/canvas, without network dependencies.
// This tests playback logic and data, not browser layout or canvas rasterization.
const fs=require('fs'),path=require('path'),vm=require('vm'),assert=require('assert');
const root=process.argv[2];
if(!root)throw new Error('Usage: node scripts/check_replays.cjs artifacts/suite_NAME');
const index=fs.readFileSync(path.join(root,'index.html'),'utf8');
for(const m of index.matchAll(/href="([^"]+)"/g))assert(fs.existsSync(path.join(root,m[1])),m[1]);
let count=0;
for(const entry of fs.readdirSync(root,{withFileTypes:true})){
 if(!entry.isDirectory())continue;
 const file=path.join(root,entry.name,'replay.html');if(!fs.existsSync(file))continue;
 const html=fs.readFileSync(file,'utf8'),script=html.match(/<script>([\s\S]*?)<\/script>/)[1];
 assert(!html.includes('__PAYLOAD__'));
 const canvas={getContext:()=>new Proxy({},{get:()=>()=>{}})};
 const slider={value:'0',max:'0'},button={textContent:''},speed={value:'1'},stats={textContent:''};
 const elements={'canvas':canvas,'#slider':slider,'#play':button,'#speed':speed,'#stats':stats,
                 '#title':{textContent:''},'#outcome':{textContent:''}};
 let callback;
 const context=vm.createContext({document:{querySelector:s=>{assert(elements[s],s);return elements[s]}},
  requestAnimationFrame:f=>{callback=f}});
 vm.runInContext(script,context,{filename:file});
 assert(stats.textContent.includes('t = 0.00'));
 assert(+slider.max>0);button.onclick();
 for(let t=0;t<=20000;t+=20)callback(t);
 assert.equal(+slider.value,+slider.max,'Playback must reach final frame');
 assert.equal(button.textContent,'播放','Must pause at end');
 slider.value=Math.floor(+slider.max/2);slider.oninput();const middle=slider.value;
 callback(20020);assert.equal(slider.value,middle,'Scrubbing pauses playback');
 speed.value='2';button.onclick();callback(20040);callback(20060);
 assert(+slider.value>+middle,'Playback must resume after scrubbing');
 button.onclick();const stopped=slider.value;callback(20100);assert.equal(slider.value,stopped);
 count++;
}
assert(count>0);console.log(`PASS: ${count} replays; links, initialization, playback, endpoint, scrubbing, speed and pause`);
