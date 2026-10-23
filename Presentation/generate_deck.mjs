import fs from 'node:fs/promises';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
const runtime=process.env.HW2_RUNTIME_NODE_MODULES ?? 'C:/Users/inpir/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules';
process.env.RUNTIME_NODE_MODULES=runtime;
const {Presentation,PresentationFile}=await import(pathToFileURL(path.join(runtime,'@oai/artifact-tool/dist/artifact_tool.mjs')).href);
const root=path.resolve('.');
const build=path.join(root,'tmp/slides'); await fs.mkdir(build,{recursive:true});
const content=JSON.parse(await fs.readFile('Presentation/content.json','utf8'));
const p=Presentation.create({slideSize:{width:1280,height:720}});
const navy='#0F4D92',ink='#182B40',gray='#5B6674';
function text(slide,value,x,y,w,h,size=26,bold=false,color=ink){
 const s=slide.shapes.add({geometry:'textbox',position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});
 s.text=value; s.text.style={fontSize:size,bold,color}; return s;
}
for(const [i,c] of content.entries()){
 const s=p.slides.add();s.background.fill='#FFFFFF';
 s.shapes.add({geometry:'rect',position:{left:0,top:0,width:1280,height:12},fill:navy,line:{fill:'none',width:0}});
 text(s,c.title,64,44,1152,98,40,true,navy);
 if(i===0){
  c.lines.forEach((line,j)=>text(s,line,64,210+j*78,1150,64,j===0?32:26,j===0));
 } else if(c.table){
  const table=s.tables.add({rows:c.table.length,columns:c.table[0].length,left:64,top:154,width:1152,height:260,columnWidths:[270,250,200,250,182],values:c.table});
  for(let r=0;r<c.table.length;r++)for(let col=0;col<c.table[0].length;col++){
    const cell=table.getCell(r,col);cell.fill=r===0?navy:r===2?'#EAF1F8':'#F7F9FB';
    cell.text.style={fontSize:23,color:r===0?'#FFFFFF':ink,bold:r===0||r===2};
  }
  c.lines.forEach((line,j)=>text(s,line,64,442+j*49,1152,48,24,false,j===2?navy:ink));
 } else if(c.image){
  if(i===6){
   s.images.add({blob:new Uint8Array(await fs.readFile(path.join(root,c.image))),contentType:'image/png',fit:'contain',position:{left:64,top:152,width:1152,height:310},alt:c.title});
   c.lines.forEach((line,j)=>text(s,line,64,478+j*49,1152,44,24));
  }else{
   s.images.add({blob:new Uint8Array(await fs.readFile(path.join(root,c.image))),contentType:'image/png',fit:'contain',position:{left:64,top:155,width:710,height:470},alt:c.title});
   c.lines.forEach((line,j)=>text(s,line,806,166+j*112,410,104,24));
  }
 }else{
   c.lines.forEach((line,j)=>text(s,line,64,174+j*90,1152,78,29));
 }
 text(s,`G04  |  ${i+1} / ${content.length}`,64,675,350,24,16,false,gray);
 s.speakerNotes.text=`Rehearsal budget: ${c.seconds} seconds.\n${c.notes}`;
 const png=await p.export({slide:s,format:'png',scale:1});
 await fs.writeFile(path.join(build,`slide-${String(i+1).padStart(2,'0')}.png`),new Uint8Array(await png.arrayBuffer()));
 console.log('Rendered slide',i+1);
}
const candidate=path.join(build,'candidate.pptx');await(await PresentationFile.exportPptx(p)).save(candidate);
const skill=process.env.HW2_PRESENTATION_SKILL_DIR ?? 'C:/Users/inpir/.codex/plugins/cache/openai-primary-runtime/presentations/26.1022.11436/skills/presentations';
const {finalizePresentation}=await import(pathToFileURL(path.join(skill,'container_tools/artifact_tool_utils.mjs')).href);
const final=path.join(build,'HW2_G04_presentation.pptx');
// Finalizer requires a new path per export. Rebuilds keep a timestamped private candidate.
await fs.mkdir(path.join(build,'final'),{recursive:true});
const finalPath=path.join(build,'final',`validated-${Date.now()}.pptx`);
const receipt=path.join(build,`validation-${Date.now()}.json`);
await finalizePresentation({workspaceDir:root,candidatePath:candidate,finalPath,
 pythonExecutable:process.env.HW2_RUNTIME_PYTHON ?? 'C:/Users/inpir/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe',
 integrityValidatorPath:path.join(skill,'container_tools/inspect_presentation_package_integrity.py'),
 layoutValidatorPath:path.join(skill,'container_tools/inspect_presentation_layout_geometry.py'),
 layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-heading-fit'],
 explicitTotalSlideCount:11,requiredNativeTableOwnerSlides:[],requiredNativeChartOwnerSlides:[],
 fontPolicy:{basis:'design',families:['Aptos']},verifyArtifactToolImport:true,
 receiptPath:receipt});
await fs.copyFile(receipt,path.join(build,'validation.json'));
await fs.copyFile(receipt,'Presentation/validation.json');
await fs.copyFile(finalPath,'HW2_G04_presentation.pptx');
await fs.copyFile(finalPath,'Presentation/HW2_G04_presentation.pptx');
console.log('Validated and exported G04 deck');
