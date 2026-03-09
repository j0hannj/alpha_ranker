"""Alpha Ranker — Desktop Application (Clean Rewrite)
Tabs: Portfolio | Rankings | Build | Projections | Backtest | Settings
Shared AI chat panel on the right side.
"""
import sys, os, threading, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import customtkinter as ctk
from tkinter import ttk, messagebox
import tkinter as tk
from core import portfolio, data, model, agent
from core.ollama_setup import (is_ollama_installed, is_ollama_running,
                                full_setup as ollama_full_setup, MODELS as OLLAMA_MODELS)
ctk.set_appearance_mode("dark"); ctk.set_default_color_theme("blue")

# ══════════════════════════════════════════════════════════════
# MAIN APP
# ══════════════════════════════════════════════════════════════
class AlphaRanker(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("MyFinancialAdvisor"); self.geometry("1500x950"); self.minsize(1200,750)
        # Set icon
        try:
            icon_path = Path(__file__).parent.parent / "icon.ico"
            if icon_path.exists(): self.iconbitmap(str(icon_path))
        except: pass
        # State
        self.model_results=None; self.feat_imp=None; self.model_info=None
        self.macro=None; self.model_state=None
        self.fx_rate=1.08; self.gbp_rate=1.16; self._portfolio_pnl=None
        portfolio.init_default_portfolio()
        # Layout: tabs left + chat right
        self.grid_columnconfigure(0,weight=1); self.grid_columnconfigure(1,weight=0)
        self.grid_rowconfigure(0,weight=1)
        self.tabs=ctk.CTkTabview(self,segmented_button_selected_color="#4f46e5")
        self.tabs.grid(row=0,column=0,sticky="nsew",padx=(10,5),pady=10)
        self._chat_w=380
        self.cpanel=ctk.CTkFrame(self,width=self._chat_w,corner_radius=10,fg_color="#0a0a0f")
        self.cpanel.grid(row=0,column=1,sticky="nsew",padx=(0,10),pady=10)
        self.cpanel.grid_propagate(False)
        # Build all
        for n in ["Portfolio","Rankings","Build","Projections","Backtest","Settings"]:
            self.tabs.add(n)
        self._init_style()
        self._init_chat()
        self._init_portfolio()
        self._init_rankings()
        self._init_build()
        self._init_projections()
        self._init_backtest()
        self._init_settings()
        # Load cache
        c=model.load_cached()
        if c:
            self.model_results=c.get("results"); self.feat_imp=c.get("feat_imp")
            self.model_info=c.get("model_info"); self.macro=c.get("macro")
        self.after(1000,self._refresh_prices)
        self.after(5000,self._morning_briefing)  # Market briefing after prices load
        api_key=portfolio.get_setting("anthropic_key")
        if not api_key and not is_ollama_running():
            self.after(2000,self._ollama_setup_dialog)

    def _init_style(self):
        s=ttk.Style(); s.theme_use("default")
        s.configure("T.Treeview",background="#18181b",foreground="#e4e4e7",
                    fieldbackground="#18181b",font=("JetBrains Mono",11),rowheight=28)
        s.configure("T.Treeview.Heading",background="#27272a",foreground="#a1a1aa",font=("",10,"bold"))
        s.map("T.Treeview",background=[("selected","#312e81")])

    def _tab(self,name): return self.tabs.tab(name)

    # ── SHARED CHAT PANEL ─────────────────────────────────────
    def _init_chat(self):
        p=self.cpanel; p.grid_columnconfigure(0,weight=1); p.grid_rowconfigure(1,weight=1)
        hdr=ctk.CTkFrame(p,fg_color="transparent",height=30)
        hdr.grid(row=0,column=0,sticky="ew",padx=8,pady=(8,2))
        ctk.CTkLabel(hdr,text="AI Advisor",font=("",13,"bold"),text_color="#818cf8").pack(side="left")
        self._eng_lbl=ctk.CTkLabel(hdr,text="",font=("",9),text_color="#52525b")
        self._eng_lbl.pack(side="left",padx=8)
        for txt,cmd in [("\u25B6",self._toggle_chat),("+",lambda:self._resize_chat(80)),
                        ("\u2212",lambda:self._resize_chat(-80))]:
            ctk.CTkButton(hdr,text=txt,width=24,height=24,font=("",10),
                          fg_color="#27272a",command=cmd).pack(side="right",padx=1)
        self.clog=ctk.CTkTextbox(p,font=("",12),fg_color="#09090b",wrap="word")
        self.clog.grid(row=1,column=0,sticky="nsew",padx=8,pady=2)
        self._update_engine()
        self.clog.insert("end","AI ready. Ask about portfolio, model, any stock.\n\n")
        qa=ctk.CTkFrame(p,fg_color="transparent"); qa.grid(row=2,column=0,sticky="ew",padx=8,pady=2)
        for t in ["Analyze portfolio","Risks?","Top picks?","Macro?"]:
            ctk.CTkButton(qa,text=t,height=22,font=("",9),fg_color="#1a1a2e",corner_radius=12,
                         command=lambda m=t:self._csend(m)).pack(side="left",padx=2)
        inp=ctk.CTkFrame(p,fg_color="transparent"); inp.grid(row=3,column=0,sticky="ew",padx=8,pady=(2,8))
        inp.grid_columnconfigure(0,weight=1)
        self.cinp=ctk.CTkEntry(inp,placeholder_text="Ask anything...",font=("",12),height=36)
        self.cinp.grid(row=0,column=0,sticky="ew",padx=(0,5))
        self.cinp.bind("<Return>",lambda e:self._csend())
        ctk.CTkButton(inp,text="Send",width=65,height=36,font=("",11,"bold"),
                      fg_color="#4f46e5",command=self._csend).grid(row=0,column=1)

    def _toggle_chat(self):
        if self.cpanel.winfo_viewable(): self.cpanel.grid_forget()
        else: self.cpanel.grid(row=0,column=1,sticky="nsew",padx=(0,10),pady=10)

    def _resize_chat(self,d):
        self._chat_w=max(250,min(700,self._chat_w+d)); self.cpanel.configure(width=self._chat_w)

    def _update_engine(self):
        _,n=agent.get_engine_status(portfolio.get_setting("anthropic_key"))
        self._eng_lbl.configure(text=n)

    def _csend(self,preset=None):
        msg=preset or self.cinp.get()
        if not msg or not msg.strip(): return
        self.cinp.delete(0,"end")
        self.clog.insert("end",f"\nYou: {msg}\n\nthinking...\n"); self.clog.see("end"); self.update()
        active=self.tabs.get()
        def _ask():
            ak=portfolio.get_setting("anthropic_key")
            pnl=self._portfolio_pnl or portfolio.compute_pnl(portfolio.get_all(),self.fx_rate,self.gbp_rate)
            r,acts=agent.chat(msg,pnl,self.model_results,self.macro,self.feat_imp,
                              self.model_info,ak or None,active,model_state=self.model_state)
            self.after(0,lambda:self._cshow(r,acts))
        threading.Thread(target=_ask,daemon=True).start()

    def _cshow(self,resp,acts):
        c=self.clog.get("1.0","end")
        if "thinking..." in c: self.clog.delete("end-2l","end-1l")
        self.clog.insert("end",f"{resp}\n\n"+"\u2500"*35+"\n"); self.clog.see("end")
        for a in acts: self._caction(a)

    def _caction(self,act):
        t=act.get("type",act[0] if isinstance(act,(list,tuple)) else "")
        args=act.get("args",act[1:] if isinstance(act,(list,tuple)) else [])
        self.clog.insert("end",f"  [Action: {t}]\n")
        if t=="refresh_prices": threading.Thread(target=self._refresh_prices,daemon=True).start()
        elif t=="run_model": self._run_model()
        elif t=="analyze" and args: self._stock_popup(args[0])
        elif t=="search_news" and args:
            def _s():
                from core.news import build_news_context
                ctx=build_news_context(args[0])
                self.after(0,lambda:self.clog.insert("end",ctx+"\n"))
            threading.Thread(target=_s,daemon=True).start()

    def _clog(self,msg):
        try: self.clog.insert("end",f"[!] {msg}\n"); self.clog.see("end")
        except: pass

    def _morning_briefing(self):
        """Auto-fetch market news and portfolio summary on startup."""
        ak=portfolio.get_setting("anthropic_key")
        has_ai=ak or agent._check_ollama()
        if not has_ai: return  # No AI engine, skip

        self.clog.insert("end","\nLoading market briefing...\n")
        self.clog.see("end")

        def _fetch():
            pnl=self._portfolio_pnl
            if not pnl:
                h=portfolio.get_all()
                pnl=portfolio.compute_pnl(h,self.fx_rate,self.gbp_rate)
            lang=portfolio.get_setting("user_language","English")
            prompt=(f"Give me a SHORT morning market briefing in {lang}. Include:\n"
                    f"1. How my portfolio is doing today (use the P&L data you have)\n"
                    f"2. Key market moves overnight (search news)\n"
                    f"3. Any news affecting my holdings specifically\n"
                    f"4. One actionable insight\n"
                    f"Keep it under 150 words. No greetings, go straight to the data.")
            resp,_=agent.chat(prompt,pnl,self.model_results,self.macro,self.feat_imp,
                              self.model_info,ak or None,"Portfolio",model_state=self.model_state)
            self.after(0,lambda:self._show_briefing(resp))

        threading.Thread(target=_fetch,daemon=True).start()

    def _show_briefing(self,resp):
        c=self.clog.get("1.0","end")
        if "Loading market briefing..." in c:
            self.clog.delete("end-2l","end-1l")
        self.clog.insert("end",f"\n--- MARKET BRIEFING ---\n{resp}\n\n"+"\u2500"*35+"\n")
        self.clog.see("end")

    # ── PORTFOLIO TAB ─────────────────────────────────────────
    def _init_portfolio(self):
        tab=self.tabs.tab("Portfolio")
        tab.grid_columnconfigure(0,weight=3); tab.grid_columnconfigure(1,weight=1)
        tab.grid_rowconfigure(1,weight=1); tab.grid_rowconfigure(2,weight=1)
        # Summary cards
        sf=ctk.CTkFrame(tab,fg_color="transparent")
        sf.grid(row=0,column=0,columnspan=2,sticky="ew",pady=(0,8))
        self.pf_lbl={}
        for i,(k,l) in enumerate([("val","Total Value"),("cost","Total Cost"),("pnl","P&L"),("pct","Performance")]):
            f=ctk.CTkFrame(sf,corner_radius=10); f.grid(row=0,column=i,padx=5,sticky="ew"); sf.grid_columnconfigure(i,weight=1)
            ctk.CTkLabel(f,text=l,font=("",11),text_color="#71717a").pack(anchor="w",padx=12,pady=(8,0))
            self.pf_lbl[k]=ctk.CTkLabel(f,text="...",font=("JetBrains Mono",22,"bold"))
            self.pf_lbl[k].pack(anchor="w",padx=12,pady=(0,8))
        # Table
        tf=ctk.CTkFrame(tab,corner_radius=10)
        tf.grid(row=1,column=0,sticky="nsew",padx=(0,8)); tf.grid_rowconfigure(1,weight=1); tf.grid_columnconfigure(0,weight=1)
        bf=ctk.CTkFrame(tf,fg_color="transparent"); bf.grid(row=0,column=0,sticky="ew",padx=10,pady=5)
        ctk.CTkLabel(bf,text="Holdings",font=("",14,"bold")).pack(side="left")
        ctk.CTkButton(bf,text="Refresh",width=80,height=26,font=("",10),
                      command=lambda:threading.Thread(target=self._refresh_prices,daemon=True).start()).pack(side="right",padx=3)
        ctk.CTkButton(bf,text="+ Add",width=70,height=26,font=("",10),fg_color="#047857",
                      command=self._add_dialog).pack(side="right",padx=3)
        ctk.CTkButton(bf,text="Delete",width=70,height=26,font=("",10),fg_color="#991b1b",
                      command=self._del_holding).pack(side="right",padx=3)
        cols=("ticker","type","qty","pru","price","value","pnl","pnl_pct")
        self.pf_tree=ttk.Treeview(tf,columns=cols,show="headings",style="T.Treeview")
        for c,h,w in zip(cols,["Ticker","Type","Qty","Cost","Price","Value","P&L","P&L%"],
                          [75,45,55,75,75,80,80,65]):
            self.pf_tree.heading(c,text=h); self.pf_tree.column(c,width=w,anchor="e" if c not in ("ticker","type") else "w")
        self.pf_tree.grid(row=1,column=0,sticky="nsew",padx=5,pady=5)
        self.pf_tree.bind("<Double-1>",self._edit_holding)
        self.pf_tree.tag_configure("pos",foreground="#34d399"); self.pf_tree.tag_configure("neg",foreground="#f87171")
        # Right: sector + projection
        rp=ctk.CTkFrame(tab,corner_radius=10); rp.grid(row=1,column=1,sticky="nsew")
        ctk.CTkLabel(rp,text="Sector Exposure",font=("",12,"bold")).pack(padx=10,pady=6)
        self.pf_sec=ctk.CTkTextbox(rp,font=("JetBrains Mono",10),state="disabled",fg_color="#09090b",height=110)
        self.pf_sec.pack(fill="x",padx=5,pady=3)
        ctk.CTkLabel(rp,text="DCA Projection",font=("",11,"bold")).pack(padx=10,pady=(8,2))
        self.pf_proj=ctk.CTkFrame(rp,fg_color="#09090b",corner_radius=8)
        self.pf_proj.pack(fill="both",expand=True,padx=5,pady=3)
        # Value chart
        cr=ctk.CTkFrame(tab,corner_radius=10); cr.grid(row=2,column=0,columnspan=2,sticky="nsew",pady=(8,0))
        ctk.CTkLabel(cr,text="Portfolio Value (1Y)",font=("",12,"bold")).pack(anchor="w",padx=10,pady=3)
        self.pf_chart=ctk.CTkFrame(cr,fg_color="#09090b",corner_radius=8,height=130)
        self.pf_chart.pack(fill="both",expand=True,padx=5,pady=3)

    def _refresh_display(self):
        h=portfolio.get_all()
        pnl=portfolio.compute_pnl(h,self.fx_rate,self.gbp_rate)
        self.pf_lbl["val"].configure(text=f"{pnl['total_value']:,.0f} EUR")
        self.pf_lbl["cost"].configure(text=f"{pnl['total_cost']:,.0f} EUR")
        c="#34d399" if pnl["total_pnl"]>=0 else "#f87171"
        self.pf_lbl["pnl"].configure(text=f"{pnl['total_pnl']:+,.0f} EUR",text_color=c)
        self.pf_lbl["pct"].configure(text=f"{pnl['total_pnl_pct']:+.1f}%",text_color=c)
        self.pf_tree.delete(*self.pf_tree.get_children())
        for h in pnl["holdings"]:
            cur="\u20ac" if h["currency"]=="EUR" else "$" if h["currency"]=="USD" else "\u00a3"
            self.pf_tree.insert("","end",iid=str(h["id"]),
                values=(h["ticker"],h["type"].upper(),h["units"],f"{h['avg_price']}{cur}",
                       f"{h.get('current_price','?')}{cur}",f"{h['value']:,.0f}\u20ac",
                       f"{h['pnl']:+,.0f}\u20ac",f"{h['pnl_pct']:+.1f}%"),
                tags=("pos" if h["pnl"]>=0 else "neg",))
        self.pf_sec.configure(state="normal"); self.pf_sec.delete("1.0","end")
        for s in pnl["sectors"]:
            bar="\u2588"*int(s["pct"]/3)
            self.pf_sec.insert("end",f"{s['sector'][:18]:<18} {s['pct']:>5.1f}% {bar}\n")
        self.pf_sec.configure(state="disabled")
        self._portfolio_pnl=pnl
        self._draw_projection(pnl["total_value"])
        self._draw_value_chart(portfolio.get_all())

    def _refresh_prices(self):
        h=portfolio.get_all()
        tickers=[h["ticker"] for h in h]
        if not tickers: return
        try:
            fx=data.fetch_fx()
            self.fx_rate=fx.get("EUR_USD",1.08)
            self.gbp_rate=1/fx.get("EUR_GBP",0.86) if fx.get("EUR_GBP") else 1.16
            prices=data.fetch_prices(tickers)
            hcur={h["ticker"]:h["currency"] for h in portfolio.get_all()}
            failed=[]
            for t,p in prices.items():
                pr=p["price"]; fc=p.get("currency","USD"); hc=hcur.get(t,"EUR")
                if fc=="USD" and hc=="EUR": pr=round(pr/self.fx_rate,2)
                elif fc=="GBp" and hc=="GBP": pr=round(pr/100,2)
                elif fc=="GBp" and hc=="EUR": pr=round(pr/100*self.gbp_rate,2)
                elif fc=="EUR" and hc=="USD": pr=round(pr*self.fx_rate,2)
                portfolio.update_price(t,pr)
            failed=[t for t in tickers if t not in prices]
            if failed:
                self.after(0,lambda:self._recover(failed))
        except Exception as e: print(f"Price error: {e}")
        self.after(0,self._refresh_display)

    def _recover(self,failed):
        def _do():
            from core.recovery import batch_recover_prices
            imap={h["ticker"]:h.get("isin") for h in portfolio.get_all() if h.get("isin")}
            rec=batch_recover_prices(failed,lambda m:self.after(0,lambda m=m:self._clog(m)),imap)
            for orig,fix in rec.items():
                if fix.get("fixed_ticker") and fix["fixed_ticker"]!=orig:
                    for h in portfolio.get_all():
                        if h["ticker"]==orig:
                            portfolio.update(h["id"],ticker=fix["fixed_ticker"],name=fix.get("name",""))
                            break
                if fix.get("price"):
                    portfolio.update_price(fix.get("fixed_ticker",orig),fix["price"])
            if rec: self.after(0,self._refresh_display)
        threading.Thread(target=_do,daemon=True).start()

    def _add_dialog(self):
        d=ctk.CTkToplevel(self); d.title("Add Position"); d.geometry("430x430")
        d.resizable(False,False); d.grab_set(); d.attributes("-topmost",True)
        ctk.CTkLabel(d,text="New Position",font=("",16,"bold")).pack(pady=(12,3))
        ctk.CTkLabel(d,text="Enter ticker (AAPL) or ISIN (US0378331005)",font=("",10),text_color="#71717a").pack(pady=(0,8))
        fr=ctk.CTkFrame(d,fg_color="transparent"); fr.pack(padx=20,fill="x")
        ent={}
        for i,(k,ph) in enumerate([("id","Ticker or ISIN"),("qty","Quantity"),("pru","Cost basis")]):
            ctk.CTkLabel(fr,text=k.capitalize(),font=("",12)).grid(row=i,column=0,sticky="w",pady=5)
            e=ctk.CTkEntry(fr,width=220,font=("JetBrains Mono",13),placeholder_text=ph)
            e.grid(row=i,column=1,pady=5,padx=(10,0)); ent[k]=e
        ctk.CTkLabel(fr,text="Type",font=("",12)).grid(row=3,column=0,sticky="w",pady=5)
        tp=ctk.CTkOptionMenu(fr,width=220,values=["stock","etf"]); tp.grid(row=3,column=1,pady=5,padx=(10,0))
        ctk.CTkLabel(fr,text="Currency",font=("",12)).grid(row=4,column=0,sticky="w",pady=5)
        cu=ctk.CTkOptionMenu(fr,width=220,values=["EUR","USD","GBP","CHF"]); cu.grid(row=4,column=1,pady=5,padx=(10,0))
        err=ctk.CTkLabel(d,text="",font=("",11),text_color="#f87171"); err.pack(pady=5)
        def _sub():
            raw=ent["id"].get().strip().upper()
            if not raw: err.configure(text="Required"); return
            try: qty=float(ent["qty"].get().replace(",","."))
            except: err.configure(text="Invalid qty"); return
            try: pru=float(ent["pru"].get().replace(",","."))
            except: err.configure(text="Invalid cost"); return
            # ISIN detection
            isin=None; ticker=raw
            if len(raw)==12 and raw[:2].isalpha():
                isin=raw
                err.configure(text="Resolving ISIN...",text_color="#fbbf24"); d.update()
                try:
                    from core.isin import resolve_isin
                    r=resolve_isin(isin)
                    if r: ticker=r["ticker"]; err.configure(text=f"-> {ticker}",text_color="#34d399")
                    else: err.configure(text="ISIN not found",text_color="#f87171"); return
                except Exception as e: err.configure(text=str(e)[:40]); return
            sj=json.dumps({"Unknown":1.0}) if tp.get()=="etf" else None
            portfolio.add(ticker,ticker,tp.get(),qty,pru,cu.get(),sectors_json=sj)
            if isin:
                for h in portfolio.get_all():
                    if h["ticker"]==ticker: portfolio.update(h["id"],isin=isin); break
            d.destroy(); self._refresh_display()
        ctk.CTkButton(d,text="Add",width=200,height=38,font=("",13,"bold"),fg_color="#047857",command=_sub).pack(pady=12)

    def _edit_holding(self,event):
        sel=self.pf_tree.selection()
        if not sel: return
        hid=int(sel[0])
        h=None
        for x in portfolio.get_all():
            if x["id"]==hid: h=x; break
        if not h: return
        d=ctk.CTkToplevel(self); d.title(f"Edit {h['ticker']}"); d.geometry("350x300")
        d.grab_set(); d.attributes("-topmost",True)
        ctk.CTkLabel(d,text=f"Edit {h['ticker']}",font=("",16,"bold")).pack(pady=12)
        fr=ctk.CTkFrame(d,fg_color="transparent"); fr.pack(padx=20,fill="x")
        ent={}
        for i,(k,v) in enumerate([("units",h["units"]),("avg_price",h["avg_price"])]):
            lbl="Quantity" if k=="units" else "Cost Basis"
            ctk.CTkLabel(fr,text=lbl,font=("",12)).grid(row=i,column=0,sticky="w",pady=8)
            e=ctk.CTkEntry(fr,width=180,font=("JetBrains Mono",13)); e.grid(row=i,column=1,pady=8,padx=(10,0))
            e.insert(0,str(v)); ent[k]=e
        def _save():
            try: q=float(ent["units"].get().replace(",",".")); p=float(ent["avg_price"].get().replace(",","."))
            except: return
            portfolio.update(hid,units=q,avg_price=p)
            d.destroy(); self._refresh_display()
        ctk.CTkButton(d,text="Save",width=160,height=36,font=("",12,"bold"),fg_color="#4f46e5",command=_save).pack(pady=15)

    def _del_holding(self):
        for s in self.pf_tree.selection(): portfolio.delete(int(s))
        self._refresh_display()

    def _draw_projection(self,val):
        try:
            import matplotlib; matplotlib.use("Agg")
            from matplotlib.figure import Figure
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
            import numpy as np
            for w in self.pf_proj.winfo_children(): w.destroy()
            dca=2200; mo=240
            etf_n={"nasdaq":0.11,"msci world":0.08,"all-world":0.08,"ftse all":0.08}
            mr=None
            if self.model_results is not None and self._portfolio_pnl:
                wp=tw=0
                for h in self._portfolio_pnl.get("holdings",[]):
                    t,w=h["ticker"],h.get("value",0)
                    if w<=0: continue
                    m=self.model_results[self.model_results["ticker"]==t]
                    if not m.empty: wp+=(m.iloc[0]["predicted_return_pct"]/100)*w; tw+=w
                    else:
                        nm=h.get("name","").lower()
                        for frag,r in etf_n.items():
                            if frag in nm: wp+=r*w; tw+=w; break
                if tw>0: mr=max(0.02,min(wp/tw,0.25))
            fig=Figure(figsize=(3,2.2),dpi=90,facecolor="#09090b")
            ax=fig.add_subplot(111); ax.set_facecolor("#09090b")
            for ret,lbl,col in [(0.06,"6%","#f8717155"),(0.08,"8%","#71717a"),(0.10,"10%","#34d39955")]:
                r=(1+ret)**(1/12)-1; v=[val]
                for _ in range(mo): v.append(v[-1]*(1+r)+dca)
                ax.plot([i/12 for i in range(mo+1)],[x/1000 for x in v],color=col,linewidth=0.8,linestyle="--",alpha=0.6,label=lbl)
            if mr:
                r=(1+mr)**(1/12)-1; v=[val]
                for _ in range(mo): v.append(v[-1]*(1+r)+dca)
                ax.plot([i/12 for i in range(mo+1)],[x/1000 for x in v],color="#818cf8",linewidth=2.5,label=f"{mr*100:.1f}% Model")
            for y in [100,500]: ax.axhline(y=y,color="#52525b",linestyle=":",linewidth=0.4,alpha=0.4)
            ax.axhline(y=1000,color="#818cf8",linestyle=":",linewidth=0.4,alpha=0.4)
            ax.set_xlabel("Years",fontsize=7,color="#71717a"); ax.set_ylabel("k EUR",fontsize=7,color="#71717a")
            ax.tick_params(colors="#52525b",labelsize=6)
            for s in ["top","right"]: ax.spines[s].set_visible(False)
            for s in ["bottom","left"]: ax.spines[s].set_color("#27272a")
            ax.legend(fontsize=5,loc="upper left",facecolor="#18181b",edgecolor="#27272a",labelcolor="#a1a1aa")
            fig.tight_layout(pad=0.5)
            c=FigureCanvasTkAgg(fig,master=self.pf_proj); c.draw(); c.get_tk_widget().pack(fill="both",expand=True)
        except: pass

    def _draw_value_chart(self,holdings):
        try:
            import matplotlib; matplotlib.use("Agg")
            from matplotlib.figure import Figure
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
            import yfinance as yf; import pandas as pd
            for w in self.pf_chart.winfo_children(): w.destroy()
            ss={}
            for h in holdings:
                try:
                    hist=yf.Ticker(h["ticker"]).history(period="1y")["Close"]
                    if hist.empty: continue
                    wt=hist*h["units"]
                    if h["currency"]=="USD": wt=wt/self.fx_rate
                    elif h["currency"]=="GBP": wt=wt*self.gbp_rate
                    ss[h["ticker"]]=wt
                except: continue
            if not ss: return
            df=pd.DataFrame(ss).ffill().bfill()
            total=df.sum(axis=1).resample("W").last().dropna()
            if len(total)<3: return
            fig=Figure(figsize=(12,1.5),dpi=90,facecolor="#09090b")
            ax=fig.add_subplot(111); ax.set_facecolor("#09090b")
            c="#34d399" if total.iloc[-1]>=total.iloc[0] else "#f87171"
            ax.plot(total.index,total.values/1000,color=c,linewidth=2)
            ax.fill_between(total.index,total.min()/1000,total.values/1000,alpha=0.1,color=c)
            ax.annotate(f"{total.iloc[-1]/1000:.1f}k",xy=(total.index[-1],total.iloc[-1]/1000),
                       fontsize=8,fontweight="bold",color=c,ha="right",va="bottom")
            ax.set_ylabel("k EUR",fontsize=7,color="#71717a")
            ax.tick_params(colors="#52525b",labelsize=6)
            for s in ["top","right"]: ax.spines[s].set_visible(False)
            for s in ["bottom","left"]: ax.spines[s].set_color("#27272a")
            fig.tight_layout(pad=0.3)
            cv=FigureCanvasTkAgg(fig,master=self.pf_chart); cv.draw(); cv.get_tk_widget().pack(fill="both",expand=True)
        except: pass

    # ── RANKINGS TAB ──────────────────────────────────────────
    def _init_rankings(self):
        tab=self.tabs.tab("Rankings"); tab.grid_columnconfigure(0,weight=1); tab.grid_rowconfigure(1,weight=1)
        top=ctk.CTkFrame(tab,fg_color="transparent"); top.grid(row=0,column=0,sticky="ew",pady=(0,6))
        ctk.CTkButton(top,text="Run Model",width=130,height=30,font=("",12,"bold"),
                      fg_color="#4f46e5",command=self._run_model).pack(side="left")
        self.rk_status=ctk.CTkLabel(top,text="Not trained",font=("",10),text_color="#71717a")
        self.rk_status.pack(side="left",padx=12)
        ff=ctk.CTkFrame(top,fg_color="transparent"); ff.pack(side="right")
        self.hz_var=ctk.StringVar(value="12")
        for m in ["3","6","12","24"]:
            ctk.CTkRadioButton(ff,text=f"{m}M",variable=self.hz_var,value=m,font=("",10),
                              command=self._upd_rankings).pack(side="left",padx=2)
        cols=("rank","ticker","name","sector","return","conviction","analyst","sentiment","pe","growth","fcf","mom")
        self.rk_tree=ttk.Treeview(tab,columns=cols,show="headings",style="T.Treeview")
        for c,h,w in zip(cols,["#","Ticker","Name","Sector","Predicted","Conv","Analyst","Sent","P/E","Grwth","FCF","Mom"],
                          [30,60,125,95,80,70,65,65,50,55,50,55]):
            self.rk_tree.heading(c,text=h); self.rk_tree.column(c,width=w,anchor="e" if c not in ("ticker","name","sector","analyst") else "w")
        self.rk_tree.grid(row=1,column=0,sticky="nsew")
        self.rk_tree.bind("<Double-1>",lambda e:self._stock_popup_from_tree())
        self.rk_tree.tag_configure("hot",foreground="#34d399",font=("JetBrains Mono",11,"bold"))
        self.rk_tree.tag_configure("warm",foreground="#fbbf24")
        self.rk_tree.tag_configure("normal",foreground="#e4e4e7")
        self.rk_tree.tag_configure("cold",foreground="#71717a")

    def _run_model(self):
        self.rk_status.configure(text="Training ensemble...",text_color="#fbbf24")
        def _train():
            def cb(m): self.after(0,lambda m=m:self.rk_status.configure(text=m[:80]))
            try:
                for k,s in [("fred_key","FRED_API_KEY"),("fmp_key","FMP_API_KEY"),("av_key","ALPHA_VANTAGE_KEY")]:
                    v=portfolio.get_setting(k)
                    if v: os.environ[s]=v
                res,fi,info,mac=model.run_full_pipeline(callback=cb)
                if res is None: self.after(0,lambda:self.rk_status.configure(text="Failed",text_color="#f87171")); return
                self.model_results=res; self.feat_imp=fi; self.model_info=info; self.macro=mac
                self.model_state=model.get_model_state()
                pmic=info.get("per_model_ic",{})
                pm=" ".join(f"{n[:3]}:{v:.3f}" for n,v in pmic.items()) if pmic else str(info.get("n_stocks","?"))+" stocks"
                self.after(0,lambda:self.rk_status.configure(text=f"IC:{info.get('spearman_rank_corr','?')} | {pm}",text_color="#34d399"))
                self.after(0,self._upd_rankings)
            except Exception as e:
                self.after(0,lambda:self.rk_status.configure(text=f"Error: {str(e)[:60]}",text_color="#f87171"))
        threading.Thread(target=_train,daemon=True).start()

    def _upd_rankings(self):
        if self.model_results is None: return
        hz=int(self.hz_var.get()); sc=lambda r: r if hz==12 else round(r*(hz/12)**0.75,1)
        self.rk_tree.delete(*self.rk_tree.get_children())
        for _,r in self.model_results.head(50).iterrows():
            ret=sc(r["predicted_return_pct"]); conf=r.get("confidence",0)
            conv=_conv(conf)
            pe=f"{r['pe_forward']:.1f}" if _ok(r.get("pe_forward")) else "-"
            gr=f"{r['revenue_growth']*100:.0f}%" if _ok(r.get("revenue_growth")) else "-"
            fcf=f"{r['fcf_yield']*100:.1f}%" if _ok(r.get("fcf_yield")) else "-"
            mom=f"{r['momentum_12_1']*100:.1f}%" if _ok(r.get("momentum_12_1")) else "-"
            reco=r.get("recommendation","")
            an={"strongBuy":"BUY++","buy":"BUY","overweight":"OW","hold":"HOLD","underweight":"UW","sell":"SELL"}.get(str(reco),"-") if _ok(reco) else "-"
            s=r.get("news_sentiment",None)
            sn="+++ Bull" if _ok(s) and s>0.3 else "+ Pos" if _ok(s) and s>0.1 else "--- Bear" if _ok(s) and s<-0.3 else "- Neg" if _ok(s) and s<-0.1 else "~ Neut" if _ok(s) else "-"
            if conf>=1.8 and ret>20: tag="hot"; rd=f">> +{ret}% <<"
            elif conf>=1.2 and ret>15: tag="warm"; rd=f"+{ret}%"
            elif conf<0.5: tag="cold"; rd=f"+{ret}%"
            else: tag="normal"; rd=f"+{ret}%"
            self.rk_tree.insert("","end",values=(int(r["rank"]),r["ticker"],r.get("name","")[:20],
                r.get("sector","")[:16],rd,conv,an,sn,pe,gr,fcf,mom),tags=(tag,))

    def _stock_popup_from_tree(self):
        sel=self.rk_tree.selection()
        if sel:
            v=self.rk_tree.item(sel[0],"values")
            if v: self._stock_popup(v[1])

    def _stock_popup(self,ticker):
        if self.model_results is None: return
        row=self.model_results[self.model_results["ticker"]==ticker]
        if row.empty: return
        r=row.iloc[0]
        d=ctk.CTkToplevel(self); d.title(ticker); d.geometry("700x550"); d.grab_set(); d.attributes("-topmost",True)
        hd=ctk.CTkFrame(d,fg_color="transparent"); hd.pack(fill="x",padx=15,pady=8)
        ctk.CTkLabel(hd,text=ticker,font=("JetBrains Mono",24,"bold")).pack(side="left")
        ctk.CTkLabel(hd,text=r.get("name",""),font=("",12),text_color="#a1a1aa").pack(side="left",padx=8)
        pred=r.get("predicted_return_pct",0); pc="#34d399" if pred>15 else "#fbbf24" if pred>5 else "#a1a1aa"
        ctk.CTkLabel(hd,text=f"+{pred}%",font=("JetBrains Mono",20,"bold"),text_color=pc).pack(side="right")
        ctk.CTkLabel(hd,text=_conv(r.get("confidence",0)),font=("",11),text_color="#fbbf24").pack(side="right",padx=10)
        # Chart
        cf=ctk.CTkFrame(d,fg_color="#09090b",corner_radius=8); cf.pack(fill="both",expand=True,padx=15,pady=3)
        try:
            import matplotlib; matplotlib.use("Agg")
            from matplotlib.figure import Figure
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
            import yfinance as yf
            hist=yf.Ticker(ticker).history(period="1y")
            if not hist.empty:
                fig=Figure(figsize=(6.5,2),dpi=100,facecolor="#09090b")
                ax=fig.add_subplot(111); ax.set_facecolor("#09090b")
                cl=hist["Close"]; c="#34d399" if cl.iloc[-1]>=cl.iloc[0] else "#f87171"
                ax.plot(cl.index,cl.values,color=c,linewidth=1.5); ax.fill_between(cl.index,cl.values,alpha=0.1,color=c)
                ax.tick_params(colors="#52525b",labelsize=7)
                for s in ["top","right"]: ax.spines[s].set_visible(False)
                for s in ["bottom","left"]: ax.spines[s].set_color("#27272a")
                fig.tight_layout(pad=0.5)
                cv=FigureCanvasTkAgg(fig,master=cf); cv.draw(); cv.get_tk_widget().pack(fill="both",expand=True)
        except: ctk.CTkLabel(cf,text="Chart N/A",text_color="#52525b").pack(pady=15)
        # Fundamentals
        gf=ctk.CTkFrame(d,corner_radius=8); gf.pack(fill="x",padx=15,pady=5)
        gg=ctk.CTkFrame(gf,fg_color="transparent"); gg.pack(fill="x",padx=8,pady=6)
        for i,(lbl,val,t) in enumerate([("P/E",r.get("pe_forward"),"r"),("Growth",r.get("revenue_growth"),"p"),
                 ("Margin",r.get("profit_margin"),"p"),("ROE",r.get("roe"),"p"),("FCF",r.get("fcf_yield"),"p"),
                 ("D/E",r.get("debt_to_equity"),"r"),("Mom12",r.get("momentum_12_1"),"p"),("Vol",r.get("volatility_12m"),"p"),
                 ("Drawdown",r.get("drawdown_from_high"),"p"),("DivYld",r.get("dividend_yield"),"p")]):
            f=ctk.CTkFrame(gg,fg_color="#09090b",corner_radius=5); f.grid(row=i//5,column=i%5,padx=2,pady=2,sticky="ew")
            gg.grid_columnconfigure(i%5,weight=1)
            ctk.CTkLabel(f,text=lbl,font=("",8),text_color="#71717a").pack(padx=4,pady=(3,0))
            txt=f"{val*100:.1f}%" if t=="p" and _ok(val) else f"{val:.1f}" if t=="r" and _ok(val) else "-"
            ctk.CTkLabel(f,text=txt,font=("JetBrains Mono",11,"bold")).pack(padx=4,pady=(0,3))

    # ── BUILD TAB (Portfolio Construction) ────────────────────
    def _init_build(self):
        tab=self.tabs.tab("Build"); tab.grid_columnconfigure(0,weight=1); tab.grid_rowconfigure(3,weight=1)
        # Row 0: Controls
        top=ctk.CTkFrame(tab,fg_color="transparent"); top.grid(row=0,column=0,sticky="ew",pady=(0,6))
        ctk.CTkLabel(top,text="Budget:",font=("",12)).pack(side="left")
        self.bld_budget=ctk.CTkEntry(top,width=90,font=("JetBrains Mono",13)); self.bld_budget.pack(side="left",padx=5)
        self.bld_budget.insert(0,"5000"); ctk.CTkLabel(top,text="EUR",font=("",11)).pack(side="left")
        ctk.CTkLabel(top,text="Fees:",font=("",11)).pack(side="left",padx=(12,4))
        self.bld_fees=ctk.CTkEntry(top,width=50,font=("JetBrains Mono",13)); self.bld_fees.pack(side="left")
        self.bld_fees.insert(0,"10"); ctk.CTkLabel(top,text="EUR/trade",font=("",10)).pack(side="left")
        ctk.CTkLabel(top,text="Max positions:",font=("",11)).pack(side="left",padx=(12,4))
        self.bld_max=ctk.CTkOptionMenu(top,width=55,values=["3","5","8","10","15"],font=("",11))
        self.bld_max.set("8"); self.bld_max.pack(side="left")
        ctk.CTkLabel(top,text="Confidence:",font=("",11)).pack(side="left",padx=(12,4))
        self.bld_confidence=ctk.CTkOptionMenu(top,width=140,values=["Low (more diversification)","Medium","High (strongest signals)"],font=("",10))
        self.bld_confidence.set("Medium"); self.bld_confidence.pack(side="left")
        ctk.CTkButton(top,text="Generate",width=120,height=30,font=("",12,"bold"),
                      fg_color="#4f46e5",command=self._gen_build).pack(side="right")

        # Row 1: ETF/Stock slider
        sl_frame=ctk.CTkFrame(tab,fg_color="transparent"); sl_frame.grid(row=1,column=0,sticky="ew",pady=(0,4))
        ctk.CTkLabel(sl_frame,text="ETF",font=("",11,"bold"),text_color="#818cf8").pack(side="left",padx=(5,0))
        self.bld_slider=ctk.CTkSlider(sl_frame,from_=0,to=100,number_of_steps=20,width=300,
                                       command=self._slider_update)
        self.bld_slider.set(30); self.bld_slider.pack(side="left",padx=8)
        ctk.CTkLabel(sl_frame,text="Stock Picks",font=("",11,"bold"),text_color="#34d399").pack(side="left")
        self.bld_slider_lbl=ctk.CTkLabel(sl_frame,text="30% ETF / 70% Stocks",font=("JetBrains Mono",11),text_color="#a1a1aa")
        self.bld_slider_lbl.pack(side="left",padx=15)
        ctk.CTkLabel(sl_frame,text="Weighting:",font=("",11)).pack(side="left",padx=(15,4))
        self.bld_weight=ctk.CTkOptionMenu(sl_frame,width=110,values=["Equal Weight","Risk Parity","Alpha Weight"],font=("",10))
        self.bld_weight.set("Alpha Weight"); self.bld_weight.pack(side="left")

        # Row 2: Two-column header
        hdr=ctk.CTkFrame(tab,fg_color="transparent"); hdr.grid(row=2,column=0,sticky="ew",pady=(4,0))
        ctk.CTkLabel(hdr,text="QUANT ENGINE",font=("",10,"bold"),text_color="#34d399").pack(side="left",padx=10)
        self.bld_ai_status=ctk.CTkLabel(hdr,text="",font=("",10),text_color="#818cf8")
        self.bld_ai_status.pack(side="left",padx=20)
        ctk.CTkButton(hdr,text="Ask AI to Adjust",width=140,height=26,font=("",10),
                      fg_color="#312e81",hover_color="#3730a3",command=self._ai_overlay).pack(side="right",padx=5)

        # Row 3: Proposal table (price, units, invested_amount, confidence, alpha_score)
        cols=("src","ticker","name","sector","alpha","conf","price","alloc","shares","cost","reason")
        self.bld_tree=ttk.Treeview(tab,columns=cols,show="headings",style="T.Treeview")
        for c,h,w in zip(cols,["Src","Ticker","Name","Sector","Alpha","Conf","Price","Invested","Qty","Cost","Reason"],
                          [35,55,110,82,58,42,58,68,38,58,180]):
            self.bld_tree.heading(c,text=h); self.bld_tree.column(c,width=w,anchor="w" if c in ("name","reason","sector") else "e")
        self.bld_tree.grid(row=3,column=0,sticky="nsew")
        self.bld_tree.tag_configure("etf",foreground="#818cf8")
        self.bld_tree.tag_configure("stock",foreground="#e4e4e7")
        self.bld_tree.tag_configure("ai",foreground="#c084fc")  # AI adjusted

        # Row 4: Bottom buttons
        bb=ctk.CTkFrame(tab,fg_color="transparent"); bb.grid(row=4,column=0,sticky="ew",pady=6)
        ctk.CTkButton(bb,text="Add All to Portfolio",width=190,height=34,font=("",12,"bold"),
                      fg_color="#047857",hover_color="#065f46",command=self._add_build_to_portfolio).pack(side="left",padx=5)
        ctk.CTkButton(bb,text="Remove Selected",width=130,height=34,font=("",11),
                      fg_color="#991b1b",command=self._remove_build_row).pack(side="left",padx=5)
        self.bld_summary=ctk.CTkLabel(bb,text="",font=("JetBrains Mono",11),text_color="#71717a")
        self.bld_summary.pack(side="right",padx=10)
        self._build_proposals=[]

    def _slider_update(self,val):
        etf_pct=int(val)
        self.bld_slider_lbl.configure(text=f"{etf_pct}% ETF / {100-etf_pct}% Stocks")

    def _gen_build(self):
        """Portfolio builder: filter by confidence, allocate with real prices, integer shares, budget cap."""
        if self.model_results is None:
            self.bld_summary.configure(text="Run the model first (Rankings tab)."); return
        try: budget=float(self.bld_budget.get().replace(",","."))
        except: budget=5000
        try: fees=float(self.bld_fees.get().replace(",","."))
        except: fees=10
        max_pos=int(self.bld_max.get())
        etf_pct=int(self.bld_slider.get())/100
        weight_map={"Equal Weight":"equal_weight","Risk Parity":"risk_parity","Alpha Weight":"alpha_weight"}
        weight_method=weight_map.get(self.bld_weight.get(),"alpha_weight")
        conf_sel=self.bld_confidence.get()
        if "Low" in conf_sel: confidence_level="LOW"
        elif "High" in conf_sel: confidence_level="HIGH"
        else: confidence_level="MEDIUM"

        from portfolio import build_suggested_portfolio
        etf_positions=[("IWDA.AS","iShares MSCI World",0.6),("VWCE.DE","Vanguard All-World",0.4)]
        raw=self.model_results.copy()
        if "current_price" not in raw.columns:
            raw["current_price"]=None
        positions=build_suggested_portfolio(
            model_results=raw,
            budget=budget,
            confidence_level=confidence_level,
            weight_method=weight_method,
            max_positions=max_pos,
            etf_budget=budget*etf_pct,
            etf_positions=etf_positions,
        )
        proposals=[]
        for p in positions:
            inv=p.get("invested_amount") or p.get("alloc") or 0
            units=p.get("units") or p.get("shares") or 0
            alpha=p.get("alpha_score")
            alpha_str=f"+{alpha*100:.1f}%" if alpha is not None else p.get("alpha_score","-")
            if isinstance(alpha_str,(int,float)): alpha_str=f"+{float(alpha_str)*100:.1f}%" if alpha_str is not None else "-"
            proposals.append({
                "src":p.get("src","?"),
                "ticker":p["ticker"],
                "name":p.get("name","")[:22],
                "sector":p.get("sector","")[:14],
                "alpha_score":alpha_str,
                "confidence":p.get("confidence"),
                "price":p.get("price"),
                "alloc":inv,
                "shares":units,
                "reason":p.get("reason","")[:40],
            })
        self._show_proposals(proposals,budget,fees)
        self.bld_ai_status.configure(text="Quant engine done. Click 'Ask AI' for adjustments.")

    def _ai_overlay(self):
        """LLM reviews the quant proposal and suggests adjustments."""
        if not self._build_proposals:
            self.bld_ai_status.configure(text="Generate a proposal first."); return
        ak=portfolio.get_setting("anthropic_key")
        if not ak and not agent._check_ollama():
            self.bld_ai_status.configure(text="No AI engine. Install one in Settings."); return

        self.bld_ai_status.configure(text="AI reviewing proposal...",text_color="#fbbf24")
        self.update()

        current_proposal=json.dumps(self._build_proposals,indent=2,default=str)
        pnl=self._portfolio_pnl or portfolio.compute_pnl(portfolio.get_all(),self.fx_rate,self.gbp_rate)
        try: budget=float(self.bld_budget.get().replace(",","."))
        except: budget=5000
        try: fees=float(self.bld_fees.get().replace(",","."))
        except: fees=10

        prompt=(f"The quantitative engine proposed this portfolio (budget {budget} EUR, fees {fees}/trade):\n"
                f"{current_proposal}\n\n"
                f"Review this proposal considering: current macro environment, news sentiment, "
                f"the user's existing portfolio, and sector risks.\n"
                f"Suggest specific adjustments: which positions to KEEP, REMOVE, or SWAP.\n"
                f"Give a short rationale for each change.\n"
                f"Then output the adjusted proposal as a JSON array with keys: "
                f"src (set to 'AI'), ticker, name, sector, alpha_score, alloc, shares, reason.\n"
                f"KEEP the total close to {budget} EUR. Output ONLY the JSON array at the end.")

        def _ask():
            resp,_=agent.chat(prompt,pnl,self.model_results,self.macro,self.feat_imp,
                              self.model_info,ak or None,"Build",model_state=self.model_state)
            self.after(0,lambda:self._parse_ai_overlay(resp,budget,fees))
        threading.Thread(target=_ask,daemon=True).start()

    def _parse_ai_overlay(self,resp,budget,fees):
        import re
        # Show AI commentary in chat panel
        text=resp.split("\n\n",1)[-1] if "\n\n" in resp else resp
        # Extract JSON
        match=re.search(r'\[.*\]',text,re.DOTALL)
        if match:
            try:
                proposals=json.loads(match.group())
                # Mark all as AI-sourced
                for p in proposals:
                    if "src" not in p: p["src"]="AI"
                self._show_proposals(proposals,budget,fees)
                self.bld_ai_status.configure(text="AI adjustments applied. Review and add to portfolio.",text_color="#c084fc")
                # Log the commentary (before JSON) in chat
                commentary=text[:text.find("[")].strip() if "[" in text else text[:300]
                if commentary:
                    self.clog.insert("end",f"\n[Build AI]\n{commentary}\n")
                    self.clog.see("end")
                return
            except: pass
        # Fallback: show response in chat, keep quant proposal
        self.clog.insert("end",f"\n[Build AI]\n{text[:500]}\n")
        self.clog.see("end")
        self.bld_ai_status.configure(text="AI review in chat panel. Quant proposal unchanged.",text_color="#fbbf24")

    def _show_proposals(self,proposals,budget,fees):
        self.bld_tree.delete(*self.bld_tree.get_children())
        self._build_proposals=proposals
        total=0; n=len(proposals)
        for p in proposals:
            tk=p.get("ticker",""); alloc=p.get("alloc",p.get("allocation_eur",0))
            shares=p.get("shares",0)
            price=p.get("price")
            if price is None and self.model_results is not None:
                m=self.model_results[self.model_results["ticker"]==tk]
                if not m.empty and _ok(m.iloc[0].get("current_price")): price=m.iloc[0]["current_price"]
            if shares==0 and alloc>0 and price and price>0: shares=int(alloc/price)
            cost=(alloc or 0)+fees; total+=cost
            alpha=p.get("alpha_score","-")
            if alpha=="-" and self.model_results is not None:
                m=self.model_results[self.model_results["ticker"]==tk]
                if not m.empty: alpha=f"+{m.iloc[0]['predicted_return_pct']:.1f}%"
            conf=p.get("confidence")
            conf_str=f"{conf:.2f}" if conf is not None and _ok(conf) else "-"
            price_str=f"{price:,.2f}" if price is not None and _ok(price) else "-"
            src=p.get("src","?")
            tag="etf" if src=="ETF" else "ai" if src=="AI" else "stock"
            self.bld_tree.insert("","end",values=(src,tk,p.get("name","")[:22],
                p.get("sector","")[:14],alpha,conf_str,price_str,f"{alloc:,.0f}",shares,f"{cost:,.0f}",
                p.get("reason","")[:40]),tags=(tag,))
        self.bld_summary.configure(
            text=f"Total: {total:,.0f} EUR ({n} trades, {n*fees:.0f} fees) | Budget: {budget:,.0f} EUR | "
                 f"Remaining: {budget-total:,.0f} EUR")

    def _add_build_to_portfolio(self):
        if not self._build_proposals: return
        added=0
        for p in self._build_proposals:
            tk=p.get("ticker",""); shares=p.get("shares",0); alloc=p.get("alloc",p.get("allocation_eur",0))
            if not tk: continue
            price=p.get("price")
            if price is None or not _ok(price):
                if self.model_results is not None:
                    match=self.model_results[self.model_results["ticker"]==tk]
                    if not match.empty and _ok(match.iloc[0].get("current_price")):
                        price=match.iloc[0]["current_price"]
            if (price is None or price<=0) and alloc>0 and shares>0: price=alloc/shares
            if price is None or price<=0: continue
            if shares<=0 and alloc>0: shares=max(1,int(alloc/price))
            existing=[h for h in portfolio.get_all() if h["ticker"]==tk]
            if existing:
                h=existing[0]; new_units=h["units"]+shares
                new_avg=(h["avg_price"]*h["units"]+price*shares)/new_units
                portfolio.update(h["id"],units=new_units,avg_price=round(new_avg,2))
            else:
                typ="etf" if p.get("src")=="ETF" or any(x in tk.upper() for x in ["IWDA","VWCE","QQQ","SPY","VTI"]) else "stock"
                portfolio.add(tk,p.get("name",tk),typ,shares,round(float(price),2),"EUR")
            added+=1
        self._refresh_display()
        self.bld_summary.configure(text=f"Added {added} positions to portfolio!")

    def _remove_build_row(self):
        sel=self.bld_tree.selection()
        for s in sel: self.bld_tree.delete(s)
        remaining=[]
        for item in self.bld_tree.get_children():
            vals=self.bld_tree.item(item,"values")
            # cols: src,ticker,name,sector,alpha,conf,price,alloc,shares,cost,reason
            try: alloc=int(float(str(vals[7]).replace(",","")))
            except: alloc=0
            try: shares=int(vals[8])
            except: shares=0
            remaining.append({"src":vals[0],"ticker":vals[1],"name":vals[2],"sector":vals[3],
                             "alpha_score":vals[4],"alloc":alloc,"shares":shares,"reason":vals[10]})
        self._build_proposals=remaining

    # ── PROJECTIONS TAB ───────────────────────────────────────
    def _init_projections(self):
        tab=self.tabs.tab("Projections"); tab.grid_columnconfigure(0,weight=1); tab.grid_rowconfigure(1,weight=1)
        top=ctk.CTkFrame(tab,fg_color="transparent"); top.grid(row=0,column=0,sticky="ew",pady=(0,6))
        ctk.CTkLabel(top,text="Forward Price Projections",font=("",14,"bold")).pack(side="left")
        ctk.CTkButton(top,text="Refresh",width=90,height=28,font=("",10),command=self._upd_proj).pack(side="right")
        cols=("ticker","name","price","units","val","3m","6m","12m","24m","ret")
        self.prj_tree=ttk.Treeview(tab,columns=cols,show="headings",style="T.Treeview")
        for c,h,w in zip(cols,["Ticker","Name","Price","Qty","Value","3M","6M","12M","24M","Ann"],
                          [60,130,70,45,75,75,75,80,80,60]):
            self.prj_tree.heading(c,text=h); self.prj_tree.column(c,width=w,anchor="e" if c not in ("ticker","name") else "w")
        self.prj_tree.grid(row=1,column=0,sticky="nsew")
        self.prj_tree.tag_configure("bull",foreground="#34d399"); self.prj_tree.tag_configure("bear",foreground="#f87171"); self.prj_tree.tag_configure("flat",foreground="#fbbf24")
        self.prj_lbl=ctk.CTkLabel(tab,text="Run model first.",font=("JetBrains Mono",11),text_color="#71717a")
        self.prj_lbl.grid(row=2,column=0,sticky="w",padx=10,pady=5)

    def _upd_proj(self):
        if self.model_results is None or self._portfolio_pnl is None:
            self.prj_lbl.configure(text="Run model first."); return
        projs=model.project_portfolio_prices(self._portfolio_pnl,self.model_results)
        self.prj_tree.delete(*self.prj_tree.get_children())
        tn=t12=0
        for p in projs:
            h3,h6=p["horizons"].get("3M",{}),p["horizons"].get("6M",{})
            h12,h24=p["horizons"].get("12M",{}),p["horizons"].get("24M",{})
            ar=p.get("annual_return",0); tag="bull" if ar>0.1 else "bear" if ar<0 else "flat"
            cu="\u20ac" if p["currency"]=="EUR" else "$" if p["currency"]=="USD" else "\u00a3"
            v=p["current_price"]*p["units"]; tn+=v; t12+=h12.get("value",v)
            self.prj_tree.insert("","end",values=(p["ticker"],p["name"][:20],f"{p['current_price']}{cu}",p["units"],f"{v:,.0f}{cu}",
                f"{h3.get('price','?')}{cu} ({h3.get('gain_pct',0):+.1f}%)",f"{h6.get('price','?')}{cu} ({h6.get('gain_pct',0):+.1f}%)",
                f"{h12.get('price','?')}{cu} ({h12.get('gain_pct',0):+.1f}%)",f"{h24.get('price','?')}{cu} ({h24.get('gain_pct',0):+.1f}%)",
                f"{ar*100:+.1f}%"),tags=(tag,))
        g=((t12/tn-1)*100) if tn>0 else 0
        self.prj_lbl.configure(text=f"12M: {t12:,.0f}EUR ({g:+.1f}%) | Now: {tn:,.0f}EUR")

    # ── BACKTEST TAB ──────────────────────────────────────────
    def _init_backtest(self):
        tab=self.tabs.tab("Backtest"); tab.grid_columnconfigure(0,weight=1)
        tab.grid_rowconfigure(2,weight=1); tab.grid_rowconfigure(3,weight=1)
        top=ctk.CTkFrame(tab,fg_color="transparent"); top.grid(row=0,column=0,sticky="ew",pady=(0,6))
        ctk.CTkLabel(top,text="Horizon:",font=("",11)).pack(side="left")
        self.bt_hz=ctk.CTkOptionMenu(top,values=["3","6","12","24"],width=60); self.bt_hz.set("12"); self.bt_hz.pack(side="left",padx=4)
        ctk.CTkLabel(top,text="From:",font=("",11)).pack(side="left",padx=(10,4))
        self.bt_yr=ctk.CTkOptionMenu(top,values=["2019","2020","2021","2022"],width=70); self.bt_yr.set("2019"); self.bt_yr.pack(side="left")
        ctk.CTkButton(top,text="Run Backtest",width=130,height=30,font=("",11,"bold"),fg_color="#4f46e5",
                      command=self._run_bt).pack(side="left",padx=12)
        self.bt_st=ctk.CTkLabel(top,text="",font=("",10),text_color="#71717a"); self.bt_st.pack(side="left")
        self.bt_cards=ctk.CTkFrame(tab,fg_color="transparent"); self.bt_cards.grid(row=1,column=0,sticky="ew",pady=(0,4))
        self.bt_chart=ctk.CTkFrame(tab,fg_color="#09090b",corner_radius=10); self.bt_chart.grid(row=2,column=0,sticky="nsew",pady=(0,4))
        ctk.CTkLabel(self.bt_chart,text="Run backtest to see equity curve",text_color="#52525b",font=("",10)).pack(pady=20)
        self.bt_log=ctk.CTkTextbox(tab,font=("JetBrains Mono",10),fg_color="#09090b",height=140)
        self.bt_log.grid(row=3,column=0,sticky="nsew")
        self.bt_log.insert("end","Walk-Forward Ensemble Backtester\n")

    def _run_bt(self):
        self.bt_st.configure(text="Running...",text_color="#fbbf24"); self.bt_log.delete("1.0","end")
        def _do():
            def cb(m): self.after(0,lambda m=m:self._btl(m))
            try:
                for k,s in [("fmp_key","FMP_API_KEY"),("fred_key","FRED_API_KEY")]:
                    v=portfolio.get_setting(k)
                    if v: os.environ[s]=v
                hz=int(self.bt_hz.get()); sy=int(self.bt_yr.get())
                cb("Fetching data..."); _,prices,yf=data.fetch_universe(); mac=data.fetch_macro()
                sm={t:f.get("sector","") for t,f in yf.items()}
                fk=os.environ.get("FMP_API_KEY")
                if fk:
                    cb("Full: FMP + ensemble walk-forward")
                    fd=model.fetch_all_fundamentals(list(yf.keys()),fk,cb)
                    wf=model.walk_forward_train(prices,fd,mac,sm,list(fd.keys()),sy,hz,cb)
                    if wf[0] is None: cb("Failed"); return
                    _,_,_,fi,om=wf
                else:
                    cb("Simple mode (no FMP)")
                    r=model.train_simple(prices,yf,mac,cb)
                    if r[0] is None: cb("Failed"); return
                    _,_,_,_,fi,om=r
                cb("\n"+"="*40+"\nRESULTS\n"+"="*40)
                for k,v in om.items():
                    if k in ("ls_returns_series","ic_series","per_model_ic","per_model","blend","caveat","mode"): continue
                    cb(f"  {k}: {v:.4f}" if isinstance(v,float) else f"  {k}: {v}")
                if "per_model_ic" in om:
                    cb("\nPer-model IC:")
                    for n,v in om["per_model_ic"].items(): cb(f"  {n}: {v:.4f}")
                self.after(0,lambda:self._bt_show(om))
                self.after(0,lambda:self.bt_st.configure(text="Done!",text_color="#34d399"))
            except Exception as e: cb(f"ERROR: {e}")
        threading.Thread(target=_do,daemon=True).start()

    def _btl(self,m): self.bt_log.insert("end",m+"\n"); self.bt_log.see("end")

    def _bt_show(self,om):
        # Cards
        for w in self.bt_cards.winfo_children(): w.destroy()
        cards=[]
        if "spearman_rank_corr" in om: v=om["spearman_rank_corr"]; cards.append(("IC",f"{v:.4f}","#34d399" if v>0.05 else "#fbbf24"))
        if "ic_ir" in om: v=om["ic_ir"]; cards.append(("IR",f"{v:.2f}","#34d399" if v>0.5 else "#fbbf24"))
        if "mean_ls_return" in om: v=om["mean_ls_return"]; cards.append(("L/S",f"{v:.1%}","#34d399" if v>0 else "#f87171"))
        if "hit_rate" in om: v=om["hit_rate"]; cards.append(("Hit",f"{v:.0%}","#34d399" if v>0.55 else "#fbbf24"))
        for i,(l,v,c) in enumerate(cards):
            self.bt_cards.grid_columnconfigure(i,weight=1)
            f=ctk.CTkFrame(self.bt_cards,corner_radius=10); f.grid(row=0,column=i,padx=4,sticky="ew")
            ctk.CTkLabel(f,text=l,font=("",9),text_color="#71717a").pack(padx=8,pady=(5,0))
            ctk.CTkLabel(f,text=v,font=("JetBrains Mono",16,"bold"),text_color=c).pack(padx=8,pady=(0,5))
        # Equity chart
        try:
            import matplotlib; matplotlib.use("Agg")
            from matplotlib.figure import Figure
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
            import numpy as np
            for w in self.bt_chart.winfo_children(): w.destroy()
            ls=om.get("ls_returns_series",[]); ic=om.get("ic_series",[])
            fig=Figure(figsize=(12,2.5),dpi=90,facecolor="#09090b")
            if ls:
                ax1=fig.add_subplot(121); ax1.set_facecolor("#09090b")
                cum=np.cumprod([1+r for r in ls])
                c="#34d399" if cum[-1]>=1 else "#f87171"
                ax1.plot(range(1,len(cum)+1),cum,color=c,linewidth=2)
                ax1.fill_between(range(1,len(cum)+1),1,cum,alpha=0.12,color=c)
                ax1.axhline(y=1,color="#52525b",linewidth=0.5,linestyle="--")
                ax1.annotate(f"{(cum[-1]-1)*100:+.1f}%",xy=(len(cum),cum[-1]),fontsize=9,fontweight="bold",color=c)
                ax1.set_title("L/S Equity",fontsize=9,color="#e4e4e7")
                ax1.tick_params(colors="#52525b",labelsize=7)
                for s in ["top","right"]: ax1.spines[s].set_visible(False)
                for s in ["bottom","left"]: ax1.spines[s].set_color("#27272a")
            if ic:
                ax2=fig.add_subplot(122); ax2.set_facecolor("#09090b")
                ax2.bar(range(1,len(ic)+1),ic,color=["#34d399" if v>0 else "#f87171" for v in ic],alpha=0.7)
                ax2.axhline(y=0,color="#52525b",linewidth=0.5)
                ax2.axhline(y=np.mean(ic),color="#fbbf24",linewidth=1,linestyle="--",alpha=0.7)
                ax2.set_title("IC/Period",fontsize=9,color="#e4e4e7")
                ax2.tick_params(colors="#52525b",labelsize=7)
                for s in ["top","right"]: ax2.spines[s].set_visible(False)
                for s in ["bottom","left"]: ax2.spines[s].set_color("#27272a")
            fig.tight_layout(pad=0.8)
            cv=FigureCanvasTkAgg(fig,master=self.bt_chart); cv.draw(); cv.get_tk_widget().pack(fill="both",expand=True)
        except: pass

    # ── SETTINGS TAB ──────────────────────────────────────────
    def _init_settings(self):
        tab=self.tabs.tab("Settings"); tab.grid_columnconfigure(1,weight=1)
        # Use a scrollable frame for all settings
        scroll=ctk.CTkScrollableFrame(tab,fg_color="transparent")
        scroll.pack(fill="both",expand=True,padx=5,pady=5)
        scroll.grid_columnconfigure(1,weight=1)
        self._sett={}
        row=0

        # API Keys
        ctk.CTkLabel(scroll,text="API Keys",font=("",15,"bold")).grid(row=row,column=0,columnspan=3,sticky="w",pady=(5,8)); row+=1
        for k,l,h in [("anthropic_key","Anthropic","console.anthropic.com"),
            ("fred_key","FRED","fred.stlouisfed.org"),("fmp_key","FMP","financialmodelingprep.com"),
            ("av_key","Alpha Vantage","alphavantage.co")]:
            ctk.CTkLabel(scroll,text=l,font=("",11)).grid(row=row,column=0,sticky="w",padx=8,pady=3)
            e=ctk.CTkEntry(scroll,width=300,font=("JetBrains Mono",10),show="*")
            e.grid(row=row,column=1,sticky="w",pady=3); e.insert(0,portfolio.get_setting(k,""))
            self._sett[k]=e
            ctk.CTkLabel(scroll,text=h,font=("",9),text_color="#52525b").grid(row=row,column=2,sticky="w",padx=8)
            row+=1

        # Investment Profile
        ctk.CTkLabel(scroll,text="Investment Profile",font=("",15,"bold")).grid(row=row,column=0,columnspan=3,sticky="w",pady=(15,8)); row+=1
        for k,l,d,hint in [
            ("monthly_dca","Monthly DCA","2200","Amount you invest each month (EUR)"),
            ("base_currency","Base Currency","EUR","EUR, USD, GBP, CHF"),
            ("risk_profile","Risk Profile","Moderate","Conservative / Moderate / Aggressive"),
            ("target_amount","Target Amount","1000000","Your wealth target (EUR)"),
            ("investment_horizon","Horizon (years)","20","Investment time horizon"),
        ]:
            ctk.CTkLabel(scroll,text=l,font=("",11)).grid(row=row,column=0,sticky="w",padx=8,pady=3)
            e=ctk.CTkEntry(scroll,width=200,font=("JetBrains Mono",10))
            e.grid(row=row,column=1,sticky="w",pady=3); e.insert(0,portfolio.get_setting(k,d))
            self._sett[k]=e
            ctk.CTkLabel(scroll,text=hint,font=("",9),text_color="#52525b").grid(row=row,column=2,sticky="w",padx=8)
            row+=1

        # Personal Info (for tax/projection context)
        ctk.CTkLabel(scroll,text="Personal Context",font=("",15,"bold")).grid(row=row,column=0,columnspan=3,sticky="w",pady=(15,8)); row+=1
        for k,l,d,hint in [
            ("user_age","Age","33","Your current age"),
            ("user_country","Country","Luxembourg","For tax rules context"),
            ("tax_notes","Tax Notes","Zero capital gains tax after 6 months holding","Specific tax rules the AI should know"),
            ("user_language","Preferred Language","French","Language for AI responses"),
            ("user_notes","Custom Notes","","Any other context for the AI (job, goals, constraints...)"),
        ]:
            ctk.CTkLabel(scroll,text=l,font=("",11)).grid(row=row,column=0,sticky="w",padx=8,pady=3)
            w=400 if k in ("tax_notes","user_notes") else 200
            e=ctk.CTkEntry(scroll,width=w,font=("JetBrains Mono",10))
            e.grid(row=row,column=1,columnspan=2 if w>200 else 1,sticky="w",pady=3)
            e.insert(0,portfolio.get_setting(k,d))
            self._sett[k]=e
            if w<=200:
                ctk.CTkLabel(scroll,text=hint,font=("",9),text_color="#52525b").grid(row=row,column=2,sticky="w",padx=8)
            row+=1

        ctk.CTkButton(scroll,text="Save All Settings",width=180,height=36,font=("",12,"bold"),
                      fg_color="#047857",command=self._save_settings).grid(row=row,column=0,columnspan=2,pady=15,sticky="w",padx=8)

    def _save_settings(self):
        for k,e in self._sett.items(): portfolio.set_setting(k,e.get())
        for k,ev in [("fred_key","FRED_API_KEY"),("fmp_key","FMP_API_KEY"),("av_key","ALPHA_VANTAGE_KEY")]:
            v=portfolio.get_setting(k)
            if v: os.environ[ev]=v
        self._update_engine()
        messagebox.showinfo("OK","Settings saved!")

    # ── OLLAMA SETUP ──────────────────────────────────────────
    def _ollama_setup_dialog(self):
        if is_ollama_installed() and is_ollama_running(): return
        d=ctk.CTkToplevel(self); d.title("Install AI"); d.geometry("480x400"); d.grab_set(); d.attributes("-topmost",True)
        ctk.CTkLabel(d,text="Install Local AI Agent",font=("",17,"bold")).pack(pady=(12,5))
        ctk.CTkLabel(d,text="Free, offline. Choose model:",font=("",11),text_color="#a1a1aa").pack(pady=(0,8))
        sel=ctk.StringVar(value="mistral-small")
        for k,info in OLLAMA_MODELS.items():
            f=ctk.CTkFrame(d,fg_color="transparent"); f.pack(fill="x",padx=20,pady=2)
            ctk.CTkRadioButton(f,text="",variable=sel,value=k,width=20).pack(side="left")
            lf=ctk.CTkFrame(f,fg_color="transparent"); lf.pack(side="left",fill="x",expand=True,padx=5)
            ctk.CTkLabel(lf,text=f'{info["name"]} ({info["size"]})',font=("",11,"bold")).pack(anchor="w")
            ctk.CTkLabel(lf,text=f'{info["desc"]} | RAM: {info["ram"]}',font=("",9),text_color="#71717a").pack(anchor="w")
        bf=ctk.CTkFrame(d,fg_color="transparent"); bf.pack(pady=10)
        def _go():
            d.destroy(); self._selected_model=sel.get(); self._install_ollama()
        ctk.CTkButton(bf,text="Install",width=140,height=34,font=("",12,"bold"),fg_color="#4f46e5",command=_go).pack(side="left",padx=4)
        ctk.CTkButton(bf,text="Later",width=100,height=34,fg_color="#27272a",command=d.destroy).pack(side="left",padx=4)

    def _install_ollama(self):
        self.clog.insert("end","\n--- Installing AI ---\n"); self.clog.see("end")
        mk=getattr(self,"_selected_model","mistral-small")
        def _do():
            def cb(m): self.after(0,lambda m=m:self.clog.insert("end",f"  {m}\n"))
            ok=ollama_full_setup(model_key=mk,callback=cb)
            if ok:
                self.after(0,lambda:self.clog.insert("end","\nAI ready!\n"))
                self.after(0,self._update_engine)
            else: self.after(0,lambda:self.clog.insert("end","\nFailed. Try ollama.com/download\n"))
        threading.Thread(target=_do,daemon=True).start()

# ── HELPERS ───────────────────────────────────────────────────
def _conv(s):
    if s>=2.5: return "V.High"
    if s>=1.8: return "High"
    if s>=1.2: return "Med+"
    if s>=0.7: return "Med"
    return "Low"

def _ok(v): return v is not None and v==v

def main(): AlphaRanker().mainloop()
if __name__=="__main__": main()
