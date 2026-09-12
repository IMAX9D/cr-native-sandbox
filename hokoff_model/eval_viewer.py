"""Read-only Tk spectator for the same native evaluation loop."""
import time
from expert_v1.tick_store_v1.schema import normalize_native_state


class EvalViewer:
    def __init__(self):
        import tkinter as tk
        self.root=tk.Tk();self.root.title('BC 新旧模型对战评估')
        self.closed=False;self.last_draw=0.;self.label=''
        self.root.protocol('WM_DELETE_WINDOW',self.close)
        self.status=tk.StringVar(value='准备对战…')
        tk.Label(self.root,textvariable=self.status,font=('',12),wraplength=540).pack()
        self.canvas=tk.Canvas(self.root,width=450,height=720,bg='#dce9cc');self.canvas.pack()
        tk.Label(self.root,text='只读观战 · 关闭窗口会中断评估，重新运行可续跑未完成场次').pack()
        self.root.update()

    def close(self):
        self.closed=True;self.root.destroy()

    def show(self,state):
        if self.closed: raise KeyboardInterrupt('spectator window closed')
        now=time.monotonic()
        if now-self.last_draw<.1:return
        self.last_draw=now
        self.root.update_idletasks();self.root.update()
        if self.closed: raise KeyboardInterrupt('spectator window closed')
        value=normalize_native_state(state);c=self.canvas;c.delete('all')
        c.create_rectangle(0,350,450,370,fill='#83bfd2',outline='')
        for x in (95,355):c.create_rectangle(x-22,346,x+22,374,fill='#bba177',outline='')
        for tower in value.towers:
            x=tower.x/18000*450;y=(1-tower.y/32000)*720
            color='#377de0' if tower.side==0 else '#d45c54'
            c.create_rectangle(x-16,y-16,x+16,y+16,fill=color)
            c.create_text(x,y-25,text=str(tower.hp))
        for entity in value.entities:
            x=entity.x/18000*450;y=(1-entity.y/32000)*720
            c.create_oval(x-6,y-6,x+6,y+6,fill='#377de0' if entity.side==0 else '#d45c54',outline='white')
            c.create_text(x,y-12,text=str(entity.card_id),font=('',7))
        self.status.set(self.label+'\nTick %d · 游戏时间 %.1fs'%(value.tick,value.tick/20))

    def finish(self,text):
        if not self.closed:
            self.status.set(text);self.root.mainloop()
