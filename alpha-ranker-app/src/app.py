"""Alpha Ranker — Desktop Application (Clean Rewrite)
Tabs: Portfolio | Rankings | Build | Projections | Backtest | Universe | Settings
Shared AI chat panel on the right side.
"""
import logging
import sys, os, threading, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import customtkinter as ctk
from tkinter import ttk, messagebox
import tkinter as tk
from core import portfolio, data, model, agent
from core.ranking_insights import add_ranking_insights
from core.api_cache import get_isin_map, get_display_id, get_stored_universe_list, set_stored_universe_list, set_isin_map
from core.ollama_setup import (is_ollama_installed, is_ollama_running,
                                full_setup as ollama_full_setup, MODELS as OLLAMA_MODELS)
logger = logging.getLogger(__name__)
try:
    from core import engine_config as _engine_cfg
except Exception as e:
    logger.warning("engine_config import failed: %s", e)
    _engine_cfg = None
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
        except Exception as e:
            logger.debug("iconbitmap failed: %s", e)
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
        for n in ["Portfolio","Rankings","Build","Projections","Backtest","Universe","Settings"]:
            self.tabs.add(n)
        self._init_style()
        self._init_chat()
        self._init_portfolio()
        self._init_rankings()
        self._init_build()
        self._init_projections()
        self._init_backtest()
        self._init_universe()
        self._init_settings()
        # Load cache
        self.all_horizon_results = {}
        c=model.load_cached()
        if c:
            self.model_results=c.get("results"); self.feat_imp=c.get("feat_imp")
            self.model_info=c.get("model_info"); self.macro=c.get("macro")
            self.all_horizon_results=c.get("all_horizons") or {}
            self.after(100,self._refresh_data_updated_label)
            self.after(150,self._upd_rankings)
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
        elif t=="fetch_large_cap_isins" and args:
            try: count=int(args[0]); count=max(2500,min(5000,count))
            except: count=4000
            def _fetch():
                for k,s in [("fmp_key","FMP_API_KEY")]:
                    v=portfolio.get_setting(k)
                    if v: os.environ[s]=v
                def cb(m): self.after(0,lambda m=m:self._clog(m))
                r=data.fetch_large_cap_isins(target=count,callback=cb)
                self.after(0,lambda:self._clog(f"ISIN: {len(r)} en base (objectif {count})."))
            threading.Thread(target=_fetch,daemon=True).start()
        elif t=="search_news" and args:
            def _s():
                from core.news import build_news_context
                ctx=build_news_context(args[0])
                self.after(0,lambda:self.clog.insert("end",ctx+"\n"))
            threading.Thread(target=_s,daemon=True).start()

    def _clog(self,msg):
        try: self.clog.insert("end",f"[!] {msg}\n"); self.clog.see("end")
        except Exception as e:
            logger.debug("_clog insert failed: %s", e)

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
        cols=("isin","ticker","type","qty","pru","price","price_updated","value","pnl","pnl_pct")
        self.pf_tree=ttk.Treeview(tf,columns=cols,show="headings",style="T.Treeview")
        for c,h,w in zip(cols,["ISIN","Ticker","Type","Qty","Cost","Price","Last update","Value","P&L","P&L%"],
                          [100,70,42,48,68,68,115,72,72,58]):
            self.pf_tree.heading(c,text=h); self.pf_tree.column(c,width=w,anchor="e" if c not in ("isin","ticker","type") else "w")
        self.pf_tree.grid(row=1,column=0,sticky="nsew",padx=5,pady=5)
        self.pf_tree.bind("<Double-1>",self._edit_holding)
        self.pf_tree.tag_configure("pos",foreground="#34d399"); self.pf_tree.tag_configure("neg",foreground="#f87171")
        # Right: sector + sell alerts + projection
        rp=ctk.CTkFrame(tab,corner_radius=10); rp.grid(row=1,column=1,sticky="nsew")
        ctk.CTkLabel(rp,text="Sector Exposure",font=("",12,"bold")).pack(padx=10,pady=6)
        self.pf_sec=ctk.CTkTextbox(rp,font=("JetBrains Mono",10),state="disabled",fg_color="#09090b",height=90)
        self.pf_sec.pack(fill="x",padx=5,pady=3)
        ctk.CTkLabel(rp,text="Sell Signals",font=("",11,"bold"),text_color="#f87171").pack(padx=10,pady=(6,2))
        pf_sell_cols=("isin","ticker","signal","urgency","strategy","pred","rank","agr","reason")
        self.pf_sell_frame=ctk.CTkFrame(rp,fg_color="transparent")
        self.pf_sell_frame.pack(fill="x",padx=5,pady=3)
        self.pf_sell_tree=ttk.Treeview(self.pf_sell_frame,columns=pf_sell_cols,show="headings",style="T.Treeview",height=6)
        for c,h,w in zip(pf_sell_cols,["ISIN","Ticker","Signal","Urgency","Strategy","Pred%","Rank","Agr%","Reason"],[95,52,48,52,72,52,48,48,120]):
            self.pf_sell_tree.heading(c,text=h); self.pf_sell_tree.column(c,width=w,anchor="w" if c in ("isin","ticker","reason") else "e")
        self.pf_sell_tree.pack(fill="x")
        self.pf_sell_tree.tag_configure("sell",foreground="#f87171"); self.pf_sell_tree.tag_configure("review",foreground="#fb923c"); self.pf_sell_tree.tag_configure("hold",foreground="#34d399")
        self.pf_sell_btns=ctk.CTkFrame(self.pf_sell_frame,fg_color="transparent")
        ctk.CTkButton(self.pf_sell_btns,text="Accept Sell",width=100,height=26,font=("",10),fg_color="#991b1b",hover_color="#7f1d1d",
                      command=self._accept_sell_from_selection).pack(side="left",padx=(0,6))
        ctk.CTkButton(self.pf_sell_btns,text="History",width=70,height=26,font=("",10),fg_color="#27272a",
                      command=self._show_trade_history).pack(side="left")
        self.pf_sell_placeholder=ctk.CTkLabel(self.pf_sell_frame,text="Sell alerts require model predictions. Run the alpha model first (Rankings \u2192 Run Model) to evaluate your holdings.",font=("",10),text_color="#71717a",wraplength=320)
        self._sell_signals=[]
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
        imap=get_isin_map()
        for h in pnl["holdings"]:
            cur="\u20ac" if h["currency"]=="EUR" else "$" if h["currency"]=="USD" else "\u00a3"
            price_ts=h.get("price_timestamp") or ""
            disp_id=h.get("isin") or get_display_id(h["ticker"],imap)
            self.pf_tree.insert("","end",iid=str(h["id"]),
                values=(disp_id,h["ticker"],h["type"].upper(),h["units"],f"{h['avg_price']}{cur}",
                       f"{h.get('current_price','?')}{cur}",price_ts[:16] if price_ts else "—",
                       f"{h['value']:,.0f}\u20ac",f"{h['pnl']:+,.0f}\u20ac",f"{h['pnl_pct']:+.1f}%"),
                tags=("pos" if h["pnl"]>=0 else "neg",))
        self.pf_sec.configure(state="normal"); self.pf_sec.delete("1.0","end")
        for s in pnl["sectors"]:
            bar="\u2588"*int(s["pct"]/3)
            self.pf_sec.insert("end",f"{s['sector'][:18]:<18} {s['pct']:>5.1f}% {bar}\n")
        self.pf_sec.configure(state="disabled")
        self._portfolio_pnl=pnl
        self._update_sell_alerts(h)
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
                ts=p.get("date") or p.get("price_timestamp")
                portfolio.update_price(t,pr,price_timestamp=ts)
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
        d=ctk.CTkToplevel(self); d.title(f"Edit {h['ticker']}"); d.geometry("380x380")
        d.grab_set(); d.attributes("-topmost",True)
        ctk.CTkLabel(d,text=f"Edit {h['ticker']}",font=("",16,"bold")).pack(pady=12)
        fr=ctk.CTkFrame(d,fg_color="transparent"); fr.pack(padx=20,fill="x")
        ent={}
        for i,(k,v) in enumerate([("units",h["units"]),("avg_price",h["avg_price"])]):
            lbl="Quantity" if k=="units" else "Cost Basis"
            ctk.CTkLabel(fr,text=lbl,font=("",12)).grid(row=i,column=0,sticky="w",pady=8)
            e=ctk.CTkEntry(fr,width=180,font=("JetBrains Mono",13)); e.grid(row=i,column=1,pady=8,padx=(10,0))
            e.insert(0,str(v)); ent[k]=e
        ctk.CTkLabel(fr,text="Strategy",font=("",12)).grid(row=2,column=0,sticky="w",pady=8)
        strat_var=ctk.StringVar(value=h.get("strategy_type") or "LONG_TERM")
        strat_menu=ctk.CTkOptionMenu(fr,width=180,values=["LONG_TERM","MEDIUM_TERM","SHORT_TERM"],variable=strat_var)
        strat_menu.grid(row=2,column=1,pady=8,padx=(10,0))
        def _save():
            try: q=float(ent["units"].get().replace(",",".")); p=float(ent["avg_price"].get().replace(",","."))
            except: return
            portfolio.update(hid,units=q,avg_price=p,strategy_type=strat_var.get())
            d.destroy(); self._refresh_display()
        ctk.CTkButton(d,text="Save",width=160,height=36,font=("",12,"bold"),fg_color="#4f46e5",command=_save).pack(pady=15)

    def _del_holding(self):
        for s in self.pf_tree.selection(): portfolio.delete(int(s))
        self._refresh_display()

    def _update_sell_alerts(self,holdings):
        try:
            mr = self.model_results
            no_model = mr is None or (getattr(mr, "empty", True) and mr.empty)
            if no_model:
                self.pf_sell_tree.pack_forget()
                self.pf_sell_btns.pack_forget()
                self.pf_sell_placeholder.pack(fill="x",pady=8,padx=4)
                self._sell_signals=[]
                return
            self.pf_sell_placeholder.pack_forget()
            self.pf_sell_tree.pack(fill="x")
            self.pf_sell_btns.pack(fill="x",pady=(2,0))
            from portfolio import get_all_sell_signals
            signals=get_all_sell_signals(holdings,self.model_results,only_open=True)
            self._sell_signals=signals
            self.pf_sell_tree.delete(*self.pf_sell_tree.get_children())
            imap=get_isin_map()
            for s in signals:
                pr=s.get("predicted_return_pct")
                pred_str=f"{pr:+.1f}%" if pr is not None and (not isinstance(pr,float) or pr==pr) else "—"
                rk=s.get("alpha_rank")
                rank_str=f"#{rk}" if rk is not None else "—"
                agr=s.get("model_agreement")
                agr_str=f"{agr*100:.0f}%" if agr is not None and (not isinstance(agr,float) or agr==agr) else "—"
                reason=(s.get("reason") or "—")
                if len(reason)>28: reason=reason[:26]+"…"
                urgency=s.get("urgency") or "—"
                tag="sell" if s.get("signal")=="SELL" else "review" if s.get("signal")=="REVIEW" else "hold"
                tk=s.get("ticker","")
                self.pf_sell_tree.insert("","end",values=(
                    get_display_id(tk,imap),
                    tk,
                    s.get("signal","HOLD"),
                    urgency,
                    (s.get("strategy") or "—")[:10],
                    pred_str,
                    rank_str,
                    agr_str,
                    reason,
                ),tags=(tag,))
        except Exception as e:
            logger.warning("_update_sell_alerts: insert row failed: %s", e)

    def _accept_sell_from_selection(self):
        sel=self.pf_sell_tree.selection()
        if not sel or not getattr(self,"_sell_signals",None): return
        idx=self.pf_sell_tree.index(sel[0])
        if idx<0 or idx>=len(self._sell_signals): return
        s=self._sell_signals[idx]
        if s.get("signal") not in ("SELL","REVIEW"): return
        ticker=s.get("ticker")
        holding=None
        for h in portfolio.get_all():
            if h.get("ticker")==ticker: holding=h; break
        if not holding: return
        self._sell_dialog(holding,s)

    def _sell_dialog(self,holding,signal_data):
        d=ctk.CTkToplevel(self); d.title(f"Sell {holding['ticker']}")
        d.geometry("420x380"); d.grab_set(); d.attributes("-topmost",True)
        ticker=holding["ticker"]
        ctk.CTkLabel(d,text=f"Sell {ticker}?",font=("",18,"bold")).pack(pady=(12,4))
        reason=signal_data.get("reason") or "Model recommendation"
        ctk.CTkLabel(d,text=reason,font=("",11),text_color="#fb923c",wraplength=380).pack(padx=15,pady=4)
        model_note=signal_data.get("model_note","")
        if model_note:
            ctk.CTkLabel(d,text=model_note,font=("",10),text_color="#a1a1aa",wraplength=380).pack(padx=15,pady=2)
        total_units=holding["units"]
        current_price=holding.get("current_price") or holding["avg_price"]
        cur="\u20ac" if holding.get("currency")=="EUR" else "$" if holding.get("currency")=="USD" else "\u00a3"
        ctk.CTkLabel(d,text=f"You hold: {total_units} units @ {current_price:.2f} {cur}",font=("JetBrains Mono",11)).pack(pady=4)
        fr=ctk.CTkFrame(d,fg_color="transparent"); fr.pack(padx=20,fill="x",pady=8)
        sell_all_var=ctk.BooleanVar(value=True)
        qty_entry=ctk.CTkEntry(fr,width=150,font=("JetBrains Mono",13),placeholder_text=str(total_units))
        def _toggle(): qty_entry.configure(state="disabled" if sell_all_var.get() else "normal")
        ctk.CTkCheckBox(fr,text="Sell all units",variable=sell_all_var,command=_toggle).pack(anchor="w")
        ctk.CTkLabel(fr,text="Or sell quantity:",font=("",11)).pack(anchor="w",pady=(8,2))
        qty_entry.pack(anchor="w"); qty_entry.configure(state="disabled")
        ctk.CTkLabel(fr,text="Sell price:",font=("",11)).pack(anchor="w",pady=(8,2))
        price_entry=ctk.CTkEntry(fr,width=150,font=("JetBrains Mono",13)); price_entry.insert(0,f"{current_price:.2f}")
        price_entry.pack(anchor="w")
        err=ctk.CTkLabel(d,text="",font=("",10),text_color="#f87171"); err.pack(pady=2)
        def _execute():
            try: sell_price=float(price_entry.get().replace(",","."))
            except ValueError: err.configure(text="Invalid sell price"); return
            if sell_all_var.get(): sell_qty=total_units
            else:
                try: sell_qty=float(qty_entry.get().replace(",","."))
                except ValueError: err.configure(text="Invalid quantity"); return
                if sell_qty<=0 or sell_qty>total_units:
                    err.configure(text=f"Quantity must be between 0 and {total_units}"); return
            self._process_sell(holding["id"],ticker,sell_qty,total_units,sell_price,holding,signal_data=signal_data,reason=reason)
            d.destroy()
            self._refresh_display()
        ctk.CTkButton(d,text="Confirm Sell",width=200,height=38,font=("",13,"bold"),fg_color="#991b1b",hover_color="#7f1d1d",command=_execute).pack(pady=12)

    def _process_sell(self,holding_id,ticker,sell_qty,total_units,sell_price,holding,reason=None,signal_data=None):
        avg_price=holding["avg_price"]
        currency=holding.get("currency","EUR")
        pnl_realized=(sell_price-avg_price)*sell_qty
        pnl_pct=((sell_price/avg_price)-1)*100 if avg_price>0 else 0
        if sell_qty>=total_units:
            portfolio.update(holding_id,status="SOLD",units=0,current_price=sell_price)
        else:
            portfolio.update(holding_id,units=total_units-sell_qty)
        portfolio.log_sell_transaction(holding_id,ticker,sell_qty,sell_price,avg_price,pnl_realized,pnl_pct,currency,reason=reason,signal_data=signal_data)

    def _show_trade_history(self):
        d=ctk.CTkToplevel(self); d.title("Trade History")
        d.geometry("720x400"); d.grab_set(); d.attributes("-topmost",True)
        ctk.CTkLabel(d,text="Trade History",font=("",16,"bold")).pack(pady=(12,8))
        cols=("isin","ticker","action","units","price","total","pnl","pnl_pct","reason","date")
        tree=ttk.Treeview(d,columns=cols,show="headings",style="T.Treeview",height=12)
        for c,h in zip(cols,["ISIN","Ticker","Action","Units","Price","Total","P&L","P&L%","Reason","Date"]):
            tree.heading(c,text=h); tree.column(c,width=95 if c=="isin" else 72 if c!="reason" else 180,anchor="e" if c not in ("isin","ticker","reason","action") else "w")
        tree.pack(fill="both",expand=True,padx=10,pady=5)
        imap=get_isin_map()
        for r in portfolio.get_trade_history(80):
            reason=(r.get("reason") or "—")[:24]+"…" if (r.get("reason") or "") and len((r.get("reason") or ""))>24 else (r.get("reason") or "—")
            date_str=(r.get("executed_at") or "—")[:16] if r.get("executed_at") else "—"
            tk=r.get("ticker","")
            tree.insert("","end",values=(
                get_display_id(tk,imap), tk, r.get("action","SELL"), r.get("units",0), r.get("price",0),
                r.get("total_amount",0), r.get("pnl_realized",0), f"{r.get('pnl_pct',0):.1f}%" if r.get("pnl_pct") is not None else "—",
                reason, date_str,
            ))
        ctk.CTkButton(d,text="Close",width=100,height=28,command=d.destroy).pack(pady=10)

    def _draw_projection(self,val):
        try:
            for w in self.pf_proj.winfo_children(): w.destroy()
            mr = self.model_results
            pnl = self._portfolio_pnl
            no_model = mr is None or (getattr(mr, "empty", True) and mr.empty)
            no_holdings = not pnl or not pnl.get("holdings")
            if no_model or no_holdings:
                msg = "DCA projection requires model predictions. Run the alpha model first (Rankings \u2192 Run Model) to generate return forecasts."
                if no_holdings and not no_model:
                    msg = "Add holdings to your portfolio to see DCA projection."
                ctk.CTkLabel(self.pf_proj, text=msg, font=("", 10), text_color="#71717a", wraplength=320).pack(padx=12, pady=24, fill="x")
                return
            projs = model.project_portfolio_prices(
                pnl, mr, model_info=getattr(self, "model_info", None),
                all_horizon_results=getattr(self, "all_horizon_results", None),
            )
            if not projs:
                ctk.CTkLabel(self.pf_proj, text="No price projections. Run the model for your holdings.", font=("", 10), text_color="#71717a", wraplength=320).pack(padx=12, pady=24, fill="x")
                return
            H = 12
            if isinstance(getattr(self, "model_info", None), dict) and self.model_info.get("prediction_horizon_months") is not None:
                H = int(self.model_info["prediction_horizon_months"])
            if H <= 0:
                H = 12
            total_value = sum(h.get("value", 0) or 0 for h in pnl.get("holdings", []))
            weighted_return = 0.0
            for proj in projs:
                v = (proj.get("current_price") or 0) * (proj.get("units") or 0)
                r_H = proj.get("annual_return", 0.08)
                annual_r = (1.0 + float(r_H)) ** (12.0 / H) - 1.0
                weighted_return += annual_r * v
            portfolio_annual_return = weighted_return / total_value if total_value > 0 else 0.08
            try:
                dca_val = float(str(portfolio.get_setting("monthly_dca", "2200")).replace(",", ".") or "2200")
            except Exception:
                dca_val = 2200
            mo = 240
            monthly_r = (1.0 + portfolio_annual_return) ** (1.0 / 12) - 1.0
            current_value = total_value
            total_contributed = total_value
            trace_val = [current_value / 1000]
            trace_contrib = [total_contributed / 1000]
            for _ in range(1, mo + 1):
                current_value = current_value * (1 + monthly_r) + dca_val
                total_contributed += dca_val
                trace_val.append(current_value / 1000)
                trace_contrib.append(total_contributed / 1000)
            import matplotlib
            matplotlib.use("Agg")
            from matplotlib.figure import Figure
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
            fig = Figure(figsize=(3, 2.2), dpi=90, facecolor="#09090b")
            ax = fig.add_subplot(111)
            ax.set_facecolor("#09090b")
            years = [i / 12 for i in range(mo + 1)]
            ax.plot(years, trace_val, color="#818cf8", linewidth=2.5, label=f"Portfolio ({portfolio_annual_return*100:.1f}%/yr)")
            ax.plot(years, trace_contrib, color="#52525b", linewidth=1, linestyle="--", alpha=0.7, label="Contributions")
            for y in [100, 500]:
                ax.axhline(y=y, color="#52525b", linestyle=":", linewidth=0.4, alpha=0.4)
            ax.axhline(y=1000, color="#818cf8", linestyle=":", linewidth=0.4, alpha=0.4)
            ax.set_xlabel("Years", fontsize=7, color="#71717a")
            ax.set_ylabel("k EUR", fontsize=7, color="#71717a")
            ax.tick_params(colors="#52525b", labelsize=6)
            for s in ["top", "right"]:
                ax.spines[s].set_visible(False)
            for s in ["bottom", "left"]:
                ax.spines[s].set_color("#27272a")
            ax.legend(fontsize=5, loc="upper left", facecolor="#18181b", edgecolor="#27272a", labelcolor="#a1a1aa")
            fig.tight_layout(pad=0.5)
            c = FigureCanvasTkAgg(fig, master=self.pf_proj)
            c.draw()
            c.get_tk_widget().pack(fill="both", expand=True)
        except Exception:
            pass

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
        except Exception as e:
            logger.warning("portfolio value chart draw failed: %s", e)

    # ── RANKINGS TAB ──────────────────────────────────────────
    def _init_rankings(self):
        tab=self.tabs.tab("Rankings"); tab.grid_columnconfigure(0,weight=1); tab.grid_rowconfigure(1,weight=1)
        top=ctk.CTkFrame(tab,fg_color="transparent"); top.grid(row=0,column=0,sticky="ew",pady=(0,6))
        ctk.CTkButton(top,text="Run Model",width=130,height=30,font=("",12,"bold"),
                      fg_color="#4f46e5",command=self._run_model).pack(side="left")
        self.rk_status=ctk.CTkLabel(top,text="Not trained",font=("",10),text_color="#71717a")
        self.rk_status.pack(side="left",padx=12)
        self.rk_data_updated=ctk.CTkLabel(top,text="",font=("",11),text_color="#34d399")
        self.rk_data_updated.pack(side="left",padx=12)
        self.rk_progress=ctk.CTkProgressBar(top,width=180,height=8); self.rk_progress.pack(side="left",padx=6); self.rk_progress.set(0)
        self.rk_progress.pack_forget()
        ff=ctk.CTkFrame(top,fg_color="transparent"); ff.pack(side="right")
        self.hz_var=ctk.StringVar(value="12")
        for m, lbl in [("3","3M"),("6","6M"),("12","12M"),("24","24M"),("120","10Y")]:
            ctk.CTkRadioButton(ff,text=lbl,variable=self.hz_var,value=m,font=("",10),
                              command=self._upd_rankings).pack(side="left",padx=2)
        ctk.CTkLabel(ff,text="Horizon = best performers for that horizon. Run model to generate all.",font=("",9),text_color="#71717a").pack(side="left",padx=(8,0))
        cols=("rank","isin","ticker","name","sector","change","stability","return","conviction","analyst","sentiment","pe","growth","fcf","mom")
        self.rk_tree=ttk.Treeview(tab,columns=cols,show="headings",style="T.Treeview")
        for c,h,w in zip(cols,["#","ISIN","Ticker","Name","Sector","Change","Stability","Predicted","Conv","Analyst","Sent","P/E","Grwth","FCF","Mom"],
                          [30,100,60,115,85,58,95,72,62,58,58,48,50,48,50]):
            self.rk_tree.heading(c,text=h); self.rk_tree.column(c,width=w,anchor="e" if c not in ("isin","ticker","name","sector","analyst","stability") else "w")
        self.rk_tree.grid(row=1,column=0,sticky="nsew")
        self.rk_tree.bind("<Double-1>",lambda e:self._stock_popup_from_tree())
        self._rk_tooltip_id=None; self._rk_tooltip_win=None
        self.rk_tree.bind("<Motion>",self._rk_on_motion); self.rk_tree.bind("<Leave>",self._rk_on_leave)
        self.rk_tree.tag_configure("hot",foreground="#34d399",font=("JetBrains Mono",11,"bold"))
        self.rk_tree.tag_configure("warm",foreground="#fbbf24")
        self.rk_tree.tag_configure("normal",foreground="#e4e4e7")
        self.rk_tree.tag_configure("cold",foreground="#71717a")
        self.rk_health_frame=ctk.CTkFrame(tab,fg_color="transparent")
        self.rk_health_frame.grid(row=2,column=0,sticky="ew",padx=10,pady=(4,0))
        self.rk_health_frame.grid_columnconfigure(0,weight=1)
        self.rk_health_lbl=ctk.CTkLabel(self.rk_health_frame,text="Run model to see health.",font=("",10),text_color="#71717a")
        self.rk_health_lbl.grid(row=0,column=0,sticky="w")
        self.rk_compare_frame=ctk.CTkFrame(tab,fg_color="transparent")
        self.rk_compare_frame.grid(row=3,column=0,sticky="ew",padx=10,pady=(4,0))
        self.rk_compare_frame.grid_columnconfigure(0,weight=1)
        ctk.CTkLabel(self.rk_compare_frame,text="Model comparison (Rank IC): ",font=("",10,"bold"),text_color="#a1a1aa").grid(row=0,column=0,sticky="w")
        self.rk_compare_lbl=ctk.CTkLabel(self.rk_compare_frame,text="Run model to see per-model IC.",font=("JetBrains Mono",9),text_color="#71717a")
        self.rk_compare_lbl.grid(row=0,column=1,sticky="w",padx=(4,0))

    def _refresh_data_updated_label(self):
        """Set Rankings tab label to last data update time. Always show when we have model data."""
        try:
            if not hasattr(self,"rk_data_updated"): return
            info = self.model_info or {}
            fresh = info.get("data_freshness") or {}
            disp = None
            if isinstance(fresh, dict):
                p = fresh.get("prices") or {}
                disp = p.get("display") if isinstance(p, dict) else None
            if disp:
                self.rk_data_updated.configure(text=f"Market data last updated: {disp}", text_color="#34d399")
            elif info.get("saved_at"):
                from datetime import datetime
                try: t = datetime.fromisoformat(str(info["saved_at"]).replace("Z","+00:00")).strftime("%Y-%m-%d %H:%M")
                except: t = str(info["saved_at"])[:16]
                self.rk_data_updated.configure(text=f"Data last updated: {t}", text_color="#34d399")
            elif self.model_results is not None and not self.model_results.empty:
                self.rk_data_updated.configure(text="Data loaded (timestamp unknown)", text_color="#71717a")
            else:
                self.rk_data_updated.configure(text="")
        except Exception:
            if hasattr(self,"rk_data_updated"): self.rk_data_updated.configure(text="")

    def _run_model(self):
        self.rk_status.configure(text="Training ensemble (all horizons)...",text_color="#fbbf24")
        if hasattr(self,"rk_progress"):
            self.rk_progress.pack(side="left",padx=6); self.rk_progress.set(0)
        def _train():
            def cb(m, progress=None):
                self.after(0,lambda m=m,p=progress:self._on_train_progress(m,p))
            try:
                for k,s in [("fred_key","FRED_API_KEY"),("fmp_key","FMP_API_KEY"),("av_key","ALPHA_VANTAGE_KEY")]:
                    v=portfolio.get_setting(k)
                    if v: os.environ[s]=v
                res,fi,info,mac,all_hr=model.run_full_pipeline(callback=cb)
                self.after(0,lambda: self._on_train_done(res, fi, info, mac, all_hr))
            except Exception as e:
                err = str(e)[:60]
                self.after(0, lambda err=err: self._on_train_error(err))
        threading.Thread(target=_train,daemon=True).start()

    def _on_train_progress(self, msg, progress=None):
        if hasattr(self,"rk_status"):
            self.rk_status.configure(text=(msg or "")[:85],text_color="#fbbf24")
        if progress is not None and hasattr(self,"rk_progress"):
            self.rk_progress.set(min(1.0, max(0.0, float(progress))))

    def _on_train_done(self, res, fi, info, mac, all_hr):
        if hasattr(self,"rk_progress"):
            self.rk_progress.set(1.0)
            self.after(500, lambda: getattr(self.rk_progress,"pack_forget",lambda:None)() if hasattr(self,"rk_progress") else None)
        if res is None:
            if hasattr(self,"rk_status"): self.rk_status.configure(text="Failed",text_color="#f87171")
            return
        self.model_results=res; self.feat_imp=fi; self.model_info=info; self.macro=mac
        self.all_horizon_results=all_hr if all_hr else {}
        self.model_state=model.get_model_state()
        self._refresh_data_updated_label()
        pmic=info.get("per_model_ic",{})
        pm=" ".join(f"{n[:3]}:{v:.3f}" for n,v in pmic.items()) if pmic else str(info.get("n_stocks","?"))+" stocks"
        n_h=len(self.all_horizon_results)
        if hasattr(self,"rk_status"):
            self.rk_status.configure(text=f"Done: {n_h} horizons | IC:{info.get('spearman_rank_corr','?')} | {pm}",text_color="#34d399")
        self.after(0,self._upd_rankings)
        self.after(0,self._refresh_display)

    def _on_train_error(self, err):
        if hasattr(self,"rk_progress"): self.rk_progress.pack_forget()
        if hasattr(self,"rk_status"): self.rk_status.configure(text=f"Error: {err}",text_color="#f87171")

    def _upd_rankings(self):
        if self.model_results is None: return
        hz=int(self.hz_var.get())
        all_hr=getattr(self,"all_horizon_results",None) or {}
        if all_hr and hz in all_hr:
            df50=all_hr[hz]["results"].head(50).copy()
            hz_lbl = "10Y" if hz == 120 else f"{hz}M"
            if hasattr(self,"rk_status"):
                self.rk_status.configure(text=f"Viewing: {hz_lbl} ranking (best performers for this horizon)", text_color="#34d399")
        else:
            df50=self.model_results.head(50).copy()
            if all_hr and not all_hr.get(hz):
                hz_lbl = "10Y" if hz == 120 else f"{hz}M"
                self.rk_status.configure(text=f"Horizon {hz_lbl}: run model to generate", text_color="#a1a1aa")
        db_path = portfolio.get_ranking_db_path() if getattr(portfolio, "get_ranking_db_path", None) else None
        if not db_path:
            try:
                _dp = getattr(portfolio, "DB_PATH", None)
                db_path = str(_dp.resolve()) if _dp else None
            except Exception:
                db_path = None
        prev = portfolio.get_latest_snapshot_before()
        history_by_ticker = {} if db_path else {t: portfolio.get_ranking_history(t, 20) for t in df50["ticker"].tolist()}
        display_df = add_ranking_insights(df50, prev, history_by_ticker, db_path=db_path, n_runs=10)
        # Snapshot is saved in model.run_full_pipeline / _run_simple after results, before cache
        self._rankings_display_df=display_df
        try:
            last_run=portfolio.get_current_run_timestamp()
            if last_run and hasattr(self,"rk_data_updated"):
                self.rk_data_updated.configure(text=f"Last model run: {last_run}",text_color="#34d399")
        except Exception as e:
            logger.debug("_upd_rankings get_current_run_timestamp: %s", e)
        if hasattr(self,"rk_health_lbl") and self.model_info:
            verdict,color,msg=model.assess_model_health(self.model_info)
            ic=self.model_info.get("mean_ic"); hr=self.model_info.get("hit_rate"); icir=self.model_info.get("ic_ir")
            def _num(x): return x is not None and (not isinstance(x,float) or x==x)
            summary="Mean IC: "+f"{ic:.3f}" if _num(ic) else "Mean IC: —"
            if _num(hr): summary+=f" | Hit rate: {hr*100:.0f}%"
            if _num(icir): summary+=f" | ICIR: {icir:.2f}"
            self.rk_health_lbl.configure(text=f"[{verdict}] {summary} — {msg}",text_color=color)
        elif hasattr(self,"rk_health_lbl"):
            self.rk_health_lbl.configure(text="Run model to see health.",text_color="#71717a")
        if hasattr(self,"rk_compare_lbl") and self.model_info:
            pm=self.model_info.get("per_model_ic") or {}
            if pm:
                txt=" | ".join(f"{n}: {v:.3f}" for n,v in sorted(pm.items(),key=lambda x:-x[1]))
                self.rk_compare_lbl.configure(text=txt[:200] if len(txt)>200 else txt,text_color="#e4e4e7")
            else:
                self.rk_compare_lbl.configure(text="No per-model IC (single run).",text_color="#71717a")
        self.rk_tree.delete(*self.rk_tree.get_children())
        imap=get_isin_map()
        for _,r in display_df.iterrows():
            raw_ret=r["predicted_return_pct"]; ret=round(float(raw_ret),1) if _ok(raw_ret) else None; conf=r.get("confidence",0)
            if isinstance(conf,float) and conf!=conf: conf=0.0
            conv=_conv(conf)
            pe=f"{r['pe_forward']:.1f}" if _ok(r.get("pe_forward")) else "-"
            gr=f"{r['revenue_growth']*100:.0f}%" if _ok(r.get("revenue_growth")) else "-"
            fcf=f"{r['fcf_yield']*100:.1f}%" if _ok(r.get("fcf_yield")) else "-"
            mom=f"{r['momentum_12_1']*100:.1f}%" if _ok(r.get("momentum_12_1")) else "-"
            reco=r.get("recommendation","")
            an={"strongBuy":"BUY++","buy":"BUY","overweight":"OW","hold":"HOLD","underweight":"UW","sell":"SELL"}.get(str(reco),"-") if _ok(reco) else "-"
            s=r.get("news_sentiment",None)
            sn="+++ Bull" if _ok(s) and s>0.3 else "+ Pos" if _ok(s) and s>0.1 else "--- Bear" if _ok(s) and s<-0.3 else "- Neg" if _ok(s) and s<-0.1 else "~ Neut" if _ok(s) else "-"
            if ret is not None and conf>=1.8 and ret>20: tag="hot"; rd=f">> +{ret}% <<"
            elif ret is not None and conf>=1.2 and ret>15: tag="warm"; rd=f"+{ret}%"
            elif conf<0.5 and ret is not None: tag="cold"; rd=f"+{ret}%"
            else: tag="normal"; rd=f"+{ret}%" if ret is not None else "—"
            rd_delta=r.get("rank_delta"); rd_val=None
            if rd_delta is not None and (not isinstance(rd_delta,float) or rd_delta==rd_delta): rd_val=int(rd_delta)
            ch_disp=f"+{rd_val}" if rd_val is not None and rd_val>0 else str(rd_val) if rd_val is not None and rd_val!=0 else "—"
            stab=r.get("movement_classification")
            if stab is None or (isinstance(stab,float) and stab!=stab): stab="—"
            else: stab=str(stab)[:14]
            name_disp = (r.get("name") or "")[:18]
            try:
                da = r.get("discovered_at")
                if da:
                    from datetime import datetime as dt
                    d = dt.fromisoformat(str(da).replace("Z",""))
                    if (dt.now() - d).days < 7:
                        name_disp = ((r.get("name") or "")[:14] + " NEW") if len((r.get("name") or "")) > 14 else ((r.get("name") or "") + " NEW")
            except Exception:
                pass
            self.rk_tree.insert("","end",values=(int(r["rank"]),get_display_id(r["ticker"],imap),r["ticker"],name_disp,
                r.get("sector","")[:14],ch_disp,stab,rd,conv,an,sn,pe,gr,fcf,mom),tags=(tag,))

    def _rk_tooltip_text(self, row_series, col_name):
        """Full value for tooltip by column. row_series is one row of _rankings_display_df."""
        if col_name=="name": return str(row_series.get("name") or "")
        if col_name=="sector": return str(row_series.get("sector") or "")
        if col_name=="change":
            rd=row_series.get("rank_delta"); v=None
            if rd is not None and (not isinstance(rd,float) or rd==rd): v=int(rd)
            ch="+"+str(v) if v is not None and v>0 else str(v) if v is not None else "—"
            base=f"Change since last run: {ch}"
            comm=row_series.get("ranking_commentary")
            return f"{base}\n\n{comm}" if comm else base
        if col_name=="stability":
            stab=row_series.get("movement_classification") or "—"
            si=row_series.get("stability_index"); si_str=f" (rank std: {si:.2f})" if si is not None and si==si and not (isinstance(si,float) and si!=si) else ""
            return f"Signal stability: {stab}{si_str}"
        if col_name=="return":
            p=row_series.get("predicted_return_pct"); return f"Predicted return: {p:.1f}%" if _ok(p) else "Predicted return: —"
        if col_name=="conviction": return f"Confidence: {row_series.get('confidence',0):.2f}"
        if col_name=="rank": return f"Rank: {int(row_series.get('rank',0))}"
        if col_name=="analyst": return f"Analyst recommendation: {row_series.get('recommendation','')}"
        if col_name in ("pe","growth","fcf","mom"):
            v=row_series.get({"pe":"pe_forward","growth":"revenue_growth","fcf":"fcf_yield","mom":"momentum_12_1"}[col_name])
            return f"{col_name}: {v}" if _ok(v) else f"{col_name}: —"
        if col_name=="sentiment": return f"News sentiment: {row_series.get('news_sentiment','')}"
        if col_name=="ranking_commentary": return str(row_series.get("ranking_commentary") or "")
        return str(row_series.get(col_name,""))

    def _rk_show_tooltip(self, item_id, col_idx):
        try:
            if self._rk_tooltip_win and self._rk_tooltip_win.winfo_exists(): self._rk_tooltip_win.destroy()
            if not hasattr(self,"_rankings_display_df") or self._rankings_display_df is None: return
            vals=self.rk_tree.item(item_id,"values")
            if not vals or col_idx<0 or col_idx>=len(vals): return
            ticker=vals[2]; cols=("rank","isin","ticker","name","sector","change","stability","return","conviction","analyst","sentiment","pe","growth","fcf","mom")
            if col_idx>=len(cols): return
            col_name=cols[col_idx]
            row=self._rankings_display_df[self._rankings_display_df["ticker"]==ticker]
            if row.empty: return
            r=row.iloc[0]; text=self._rk_tooltip_text(r,col_name)
            if not text: return
            self._rk_tooltip_win=ctk.CTkToplevel(self); self._rk_tooltip_win.wm_overrideredirect(True)
            self._rk_tooltip_win.wm_geometry(f"+{self.winfo_pointerx()+12}+{self.winfo_pointery()+12}")
            lbl=ctk.CTkLabel(self._rk_tooltip_win,text=text,font=("",10),wraplength=320,fg_color="#27272a",corner_radius=6,padx=10,pady=8)
            lbl.pack(); self._rk_tooltip_win.lift()
        except Exception as e:
            logger.debug("_rk_show_tooltip: %s", e)

    def _rk_on_motion(self, event):
        reg=self.rk_tree.identify_region(event.x,event.y)
        if reg!="cell":
            if self._rk_tooltip_id: self.after_cancel(self._rk_tooltip_id); self._rk_tooltip_id=None
            if self._rk_tooltip_win and self._rk_tooltip_win.winfo_exists(): self._rk_tooltip_win.destroy(); self._rk_tooltip_win=None
            return
        item=self.rk_tree.identify_row(event.y); col=self.rk_tree.identify_column(event.x)
        if not item or not col or col=="#0": return
        try: col_idx=int(col[1:],10)-1
        except (ValueError, TypeError) as e:
            logger.debug("_rk_on_motion col parse: %s", e)
            return
        if self._rk_tooltip_id: self.after_cancel(self._rk_tooltip_id)
        def _show(): self._rk_show_tooltip(item,col_idx); self._rk_tooltip_id=None
        self._rk_tooltip_id=self.after(600,_show)

    def _rk_on_leave(self, event):
        if self._rk_tooltip_id: self.after_cancel(self._rk_tooltip_id); self._rk_tooltip_id=None
        if self._rk_tooltip_win and self._rk_tooltip_win.winfo_exists(): self._rk_tooltip_win.destroy(); self._rk_tooltip_win=None

    def _stock_popup_from_tree(self):
        sel=self.rk_tree.selection()
        if sel:
            v=self.rk_tree.item(sel[0],"values")
            # v = (rank, isin, ticker, name, ...); popup needs ticker for API
            if v and len(v)>2: self._stock_popup(v[2])

    def _stock_popup(self,ticker):
        if self.model_results is None: return
        row=self.model_results[self.model_results["ticker"]==ticker]
        if row.empty: return
        r=row.iloc[0]
        disp=get_display_id(ticker)
        d=ctk.CTkToplevel(self); d.title(disp); d.geometry("700x550"); d.grab_set(); d.attributes("-topmost",True)
        hd=ctk.CTkFrame(d,fg_color="transparent"); hd.pack(fill="x",padx=15,pady=8)
        ctk.CTkLabel(hd,text=disp,font=("JetBrains Mono",24,"bold")).pack(side="left")
        ctk.CTkLabel(hd,text=f"  ({ticker})",font=("",12),text_color="#71717a").pack(side="left")
        ctk.CTkLabel(hd,text=r.get("name",""),font=("",12),text_color="#a1a1aa").pack(side="left",padx=4)
        pred=r.get("predicted_return_pct",0); pred_ok=pred==pred and pred is not None
        pc="#34d399" if pred_ok and pred>15 else "#fbbf24" if pred_ok and pred>5 else "#a1a1aa"
        ctk.CTkLabel(hd,text=f"+{pred:.1f}%" if pred_ok else "—",font=("JetBrains Mono",20,"bold"),text_color=pc).pack(side="right")
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
        ctk.CTkLabel(sl_frame,text="Strategy override:",font=("",11)).pack(side="left",padx=(15,4))
        self.bld_strategy_override=ctk.CTkOptionMenu(sl_frame,width=110,values=["Auto","LONG_TERM","MEDIUM_TERM","SHORT_TERM","DONT_SELL"],font=("",10))
        self.bld_strategy_override.set("Auto"); self.bld_strategy_override.pack(side="left")

        # Row 2: Two-column header
        hdr=ctk.CTkFrame(tab,fg_color="transparent"); hdr.grid(row=2,column=0,sticky="ew",pady=(4,0))
        ctk.CTkLabel(hdr,text="QUANT ENGINE",font=("",10,"bold"),text_color="#34d399").pack(side="left",padx=10)
        self.bld_ai_status=ctk.CTkLabel(hdr,text="",font=("",10),text_color="#818cf8")
        self.bld_ai_status.pack(side="left",padx=20)
        self.bld_data_as_of=ctk.CTkLabel(hdr,text="",font=("",9),text_color="#52525b")
        self.bld_data_as_of.pack(side="left",padx=8)
        ctk.CTkButton(hdr,text="Ask AI to Adjust",width=140,height=26,font=("",10),
                      fg_color="#312e81",hover_color="#3730a3",command=self._ai_overlay).pack(side="right",padx=5)

        # Row 3: Proposal table (price, units, invested_amount, confidence, alpha_score, holding_period, target, stop_loss, consensus)
        cols=("src","isin","ticker","name","sector","alpha","conf","price","alloc","shares","horizon","target","stop","consensus","reason")
        self.bld_tree=ttk.Treeview(tab,columns=cols,show="headings",style="T.Treeview")
        for c,h,w in zip(cols,["Src","ISIN","Ticker","Name","Sector","Alpha","Conf","Price","Invested","Qty","Horizon","Target","Stop","Consensus","Reason"],
                          [35,95,55,100,70,50,40,52,62,36,52,52,52,52,140]):
            self.bld_tree.heading(c,text=h); self.bld_tree.column(c,width=w,anchor="w" if c in ("isin","name","reason","sector") else "e")
        self.bld_tree.grid(row=3,column=0,sticky="nsew")
        self.bld_tree.tag_configure("etf",foreground="#818cf8")
        self.bld_tree.tag_configure("stock",foreground="#e4e4e7")
        self.bld_tree.tag_configure("ai",foreground="#c084fc")
        self.bld_tree.tag_configure("neg",foreground="#f87171")  # SELL

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
        from portfolio.transaction_cost_model import TransactionCostParams
        broker_fee=float(portfolio.get_setting("tx_cost_broker_fee","0").replace(",",".") or "0")
        spread_bps=float(portfolio.get_setting("tx_cost_spread_bps","10").replace(",",".") or "10")
        slippage_bps=float(portfolio.get_setting("tx_cost_slippage_bps","5").replace(",",".") or "5")
        # Use selected ranking horizon so proposals match 3M/6M/12M/24M/10Y (not always 365d)
        hv = getattr(self, "hz_var", None)
        hz = int(hv.get()) if hv else 12
        horizon_months_to_days = {3: 90, 6: 180, 12: 365, 24: 730, 120: 3650}
        horizon_days = horizon_months_to_days.get(hz)
        if horizon_days is None:
            hd = portfolio.get_setting("default_holding_horizon_days")
            horizon_days = int(hd) if hd else 365
        tx_params=TransactionCostParams(broker_fee=broker_fee,spread_bps=spread_bps,slippage_bps=slippage_bps)
        etf_positions=None
        # Use model results for selected horizon when multi-horizon is available
        all_hr = getattr(self,"all_horizon_results",None) or {}
        if all_hr and hz in all_hr:
            raw = all_hr[hz]["results"].copy()
        else:
            raw = self.model_results.copy()
        if "current_price" not in raw.columns:
            raw["current_price"]=None
        # If no prices available, fetch live so build can propose candidates
        if raw["current_price"].isna().all() or (raw["current_price"]<=0).all():
            try:
                tickers=raw["ticker"].dropna().unique().tolist()[:80]
                if tickers:
                    prices=data.fetch_prices(tickers)
                    raw["current_price"]=raw["ticker"].map(lambda t: prices.get(t,{}).get("price"))
            except Exception as e:
                logger.warning("Build: fetch_prices for current_price failed: %s", e)
        positions=build_suggested_portfolio(
            model_results=raw,
            budget=budget,
            confidence_level=confidence_level,
            weight_method=weight_method,
            max_positions=max_pos,
            etf_budget=budget*etf_pct,
            etf_positions=etf_positions,
            transaction_cost_params=tx_params,
            holding_horizon_days=horizon_days,
            existing_holdings=portfolio.get_all(),
        )
        proposals=[]
        for p in positions:
            inv=p.get("invested_amount") or p.get("alloc") or 0
            units=p.get("units") or p.get("shares") or 0
            alpha=p.get("alpha_score")
            alpha_str=f"+{alpha*100:.1f}%" if alpha is not None else p.get("alpha_score","-")
            if isinstance(alpha_str,(int,float)): alpha_str=f"+{float(alpha_str)*100:.1f}%" if alpha_str is not None else "-"
            strat_override=self.bld_strategy_override.get() if hasattr(self,"bld_strategy_override") else "Auto"
            strat=p.get("strategy_type") or ("LONG_TERM" if p.get("src")=="ETF" else "SHORT_TERM")
            if strat_override and str(strat_override) != "Auto":
                strat=str(strat_override)
            holding_days=p.get("expected_holding_period") or p.get("holding_horizon") or horizon_days
            action=p.get("action") or "BUY"
            proposals.append({
                "action":action,
                "src": "SELL" if action=="SELL" else p.get("src","?"),
                "ticker":p["ticker"],
                "name":p.get("name","")[:22],
                "sector":p.get("sector","")[:14],
                "alpha_score":alpha_str,
                "alpha_score_num":p.get("alpha_score"),
                "confidence":p.get("confidence"),
                "price":p.get("price") or p.get("current_price"),
                "alloc":inv,
                "shares":units,
                "reason":p.get("reason","")[:40],
                "expected_return":p.get("expected_return"),
                "transaction_cost":p.get("transaction_cost"),
                "target_price":p.get("target_price"),
                "stop_loss":p.get("stop_loss"),
                "holding_horizon":p.get("holding_horizon",horizon_days),
                "expected_holding_period":holding_days,
                "strategy_type":strat,
                "review_date":p.get("review_date"),
                "model_consensus_score":p.get("model_consensus_score"),
            })
        self._show_proposals(proposals,budget,fees)
        self.bld_ai_status.configure(text="Quant engine done. Click 'Ask AI' for adjustments.")
        info=self.model_info or {}
        fresh=info.get("data_freshness") or {}
        disp=fresh.get("prices",{}).get("display") if isinstance(fresh.get("prices"),dict) else None
        if disp: self.bld_data_as_of.configure(text=f"Data as of: {disp}")
        elif info.get("saved_at"): self.bld_data_as_of.configure(text=f"Data as of: {str(info['saved_at'])[:16]}")
        else: self.bld_data_as_of.configure(text="")

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
            except Exception as e:
                logger.warning("Build AI parse proposals from response: %s", e)
        # Fallback: show response in chat, keep quant proposal
        self.clog.insert("end",f"\n[Build AI]\n{text[:500]}\n")
        self.clog.see("end")
        self.bld_ai_status.configure(text="AI review in chat panel. Quant proposal unchanged.",text_color="#fbbf24")

    def _show_proposals(self,proposals,budget,fees):
        self.bld_tree.delete(*self.bld_tree.get_children())
        self._build_proposals=proposals
        imap=get_isin_map()
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
            horizon=p.get("expected_holding_period") or p.get("holding_horizon")
            horizon_str=f"{horizon}d" if horizon is not None else "-"
            target=p.get("target_price"); target_str=f"{target:,.2f}" if target is not None and _ok(target) else "-"
            stop=p.get("stop_loss"); stop_str=f"{stop:,.2f}" if stop is not None and _ok(stop) else "-"
            consensus=p.get("model_consensus_score"); consensus_str=f"{consensus:.2f}" if consensus is not None and _ok(consensus) else "-"
            src=p.get("src","?")
            tag="etf" if src=="ETF" else "neg" if src=="SELL" else "ai" if src=="AI" else "stock"
            self.bld_tree.insert("","end",values=(src,get_display_id(tk,imap),tk,p.get("name","")[:22],
                p.get("sector","")[:14],alpha,conf_str,price_str,f"{alloc:,.0f}",shares,horizon_str,target_str,stop_str,consensus_str,
                p.get("reason","")[:40]),tags=(tag,))
        self.bld_summary.configure(
            text=f"Total: {total:,.0f} EUR ({n} trades, {n*fees:.0f} fees) | Budget: {budget:,.0f} EUR | "
                 f"Remaining: {budget-total:,.0f} EUR")

    def _add_build_to_portfolio(self):
        if not self._build_proposals: return
        added=0
        for p in self._build_proposals:
            if p.get("action")=="SELL": continue
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
                strat=p.get("strategy_type") or ("LONG_TERM" if typ=="etf" else "SHORT_TERM")
                holding_days=p.get("expected_holding_period") or p.get("holding_horizon")
                portfolio.add(tk,p.get("name",tk),typ,shares,round(float(price),2),"EUR",
                    strategy_type=strat,
                    confidence=p.get("confidence"),
                    alpha_score=p.get("alpha_score_num") if isinstance(p.get("alpha_score_num"),(int,float)) else None,
                    expected_return=p.get("expected_return"),
                    target_price=p.get("target_price"),
                    stop_loss=p.get("stop_loss"),
                    holding_horizon_days=holding_days,
                    transaction_cost=p.get("transaction_cost"),
                    review_date=p.get("review_date"))
            added+=1
        self._refresh_display()
        self.bld_summary.configure(text=f"Added {added} positions to portfolio!")

    def _remove_build_row(self):
        sel=self.bld_tree.selection()
        for s in sel: self.bld_tree.delete(s)
        remaining=[]
        for item in self.bld_tree.get_children():
            vals=self.bld_tree.item(item,"values")
            # cols: src,isin,ticker,name,sector,alpha,conf,price,alloc,shares,horizon,target,stop,consensus,reason
            try: alloc=int(float(str(vals[8]).replace(",","")))
            except: alloc=0
            try: shares=int(vals[9])
            except: shares=0
            remaining.append({"src":vals[0],"ticker":vals[2],"name":vals[3],"sector":vals[4],
                             "alpha_score":vals[5],"alloc":alloc,"shares":shares,"reason":vals[14]})
        self._build_proposals=remaining

    # ── PROJECTIONS TAB ───────────────────────────────────────
    def _init_projections(self):
        tab=self.tabs.tab("Projections"); tab.grid_columnconfigure(0,weight=1); tab.grid_rowconfigure(1,weight=1)
        top=ctk.CTkFrame(tab,fg_color="transparent"); top.grid(row=0,column=0,sticky="ew",pady=(0,6))
        ctk.CTkLabel(top,text="Forward Price Projections",font=("",14,"bold")).pack(side="left")
        ctk.CTkButton(top,text="Refresh",width=90,height=28,font=("",10),command=self._upd_proj).pack(side="right")
        cols=("isin","ticker","name","price","units","val","3m","6m","12m","24m","10y","ret")
        self.prj_tree=ttk.Treeview(tab,columns=cols,show="headings",style="T.Treeview")
        for c,h,w in zip(cols,["ISIN","Ticker","Name","Price","Qty","Value","3M","6M","12M","24M","10Y","Model ret"],
                          [95,60,120,65,42,70,68,68,72,72,72,68]):
            self.prj_tree.heading(c,text=h); self.prj_tree.column(c,width=w,anchor="e" if c not in ("isin","ticker","name") else "w")
        self.prj_tree.grid(row=1,column=0,sticky="nsew")
        self.prj_tree.tag_configure("bull",foreground="#34d399"); self.prj_tree.tag_configure("bear",foreground="#f87171"); self.prj_tree.tag_configure("flat",foreground="#fbbf24")
        self.prj_lbl=ctk.CTkLabel(tab,text="Run model first.",font=("JetBrains Mono",11),text_color="#71717a")
        self.prj_lbl.grid(row=2,column=0,sticky="w",padx=10,pady=5)

    def _upd_proj(self):
        if self.model_results is None or self._portfolio_pnl is None:
            self.prj_lbl.configure(text="Run model first."); return
        projs = model.project_portfolio_prices(
            self._portfolio_pnl, self.model_results, model_info=self.model_info,
            all_horizon_results=getattr(self, "all_horizon_results", None),
        )
        self.prj_tree.delete(*self.prj_tree.get_children())
        imap=get_isin_map()
        tn=t12=0
        H = 12
        if isinstance(self.model_info, dict) and self.model_info.get("prediction_horizon_months") is not None:
            H = int(self.model_info["prediction_horizon_months"])
        for p in projs:
            h3,h6=p["horizons"].get("3M",{}),p["horizons"].get("6M",{})
            h12,h24=p["horizons"].get("12M",{}),p["horizons"].get("24M",{})
            h10y=p["horizons"].get("10Y",{})
            ar=p.get("annual_return",0); tag="bull" if ar>0.1 else "bear" if ar<0 else "flat"
            cu="\u20ac" if p["currency"]=="EUR" else "$" if p["currency"]=="USD" else "\u00a3"
            v=p["current_price"]*p["units"]; tn+=v; t12+=h12.get("value",v)
            self.prj_tree.insert("","end",values=(get_display_id(p["ticker"],imap),p["ticker"],p["name"][:20],f"{p['current_price']}{cu}",p["units"],f"{v:,.0f}{cu}",
                f"{h3.get('price','?')}{cu} ({h3.get('gain_pct',0):+.1f}%)",f"{h6.get('price','?')}{cu} ({h6.get('gain_pct',0):+.1f}%)",
                f"{h12.get('price','?')}{cu} ({h12.get('gain_pct',0):+.1f}%)",f"{h24.get('price','?')}{cu} ({h24.get('gain_pct',0):+.1f}%)",
                f"{h10y.get('price','?')}{cu} ({h10y.get('gain_pct',0):+.1f}%)",f"{ar*100:+.1f}%"),tags=(tag,))
        g=((t12/tn-1)*100) if tn>0 else 0
        self.prj_lbl.configure(text=f"12M: {t12:,.0f}EUR ({g:+.1f}%) | Now: {tn:,.0f}EUR | Model horizon: {H}M (projections compounded from {H}M return)")

    # ── BACKTEST TAB ──────────────────────────────────────────
    def _init_backtest(self):
        tab=self.tabs.tab("Backtest"); tab.grid_columnconfigure(0,weight=1)
        tab.grid_rowconfigure(2,weight=1); tab.grid_rowconfigure(3,weight=1)
        top=ctk.CTkFrame(tab,fg_color="transparent"); top.grid(row=0,column=0,sticky="ew",pady=(0,6))
        ctk.CTkLabel(top,text="Horizon:",font=("",11)).pack(side="left")
        self.bt_hz=ctk.CTkOptionMenu(top,values=["3","6","12","24","120"],width=60); self.bt_hz.set("12"); self.bt_hz.pack(side="left",padx=4)
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
                cb("Fetching data...")
                # Pour le backtest, on veut les prix + infos de base pour un univers large
                alldata = data.fetch_all_data(years=hz, callback=cb)
                tickers = alldata.get("tickers") or []
                prices = alldata.get("prices")
                yf = alldata.get("fundamentals") or {}
                mac=data.fetch_macro()
                # Modélisation : uniquement titres avec ticker ET ISIN
                try:
                    import pandas as pd
                    imap=get_isin_map()
                    tickers_ok=[t for t in yf if imap.get(t)]
                    if tickers_ok and len(tickers_ok)<len(yf):
                        cb(f"Modélisation: {len(tickers_ok)}/{len(yf)} titres avec ISIN")
                        yf={t:yf[t] for t in tickers_ok if t in yf}
                        if hasattr(prices,"columns") and isinstance(prices.columns,pd.MultiIndex):
                            keep=[c for c in prices.columns if isinstance(c,tuple) and len(c)==2 and c[0] in tickers_ok]
                            if keep: prices=prices[keep].copy()
                except Exception as e:
                    logger.warning("_run_model: ISIN filter / prices trim failed: %s", e)
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
        except Exception as e:
            logger.warning("Backtest chart draw failed: %s", e)

    # ── UNIVERSE TAB (tickers connus + historique au clic) ──────
    def _init_universe(self):
        tab=self.tabs.tab("Universe"); tab.grid_columnconfigure(0,weight=0); tab.grid_columnconfigure(1,weight=1)
        tab.grid_rowconfigure(2,weight=1)
        top=ctk.CTkFrame(tab,fg_color="transparent"); top.grid(row=0,column=0,columnspan=2,sticky="ew",pady=(0,6))
        top.grid_columnconfigure(0,weight=1)
        ctk.CTkLabel(top,text="Universe — Tickers connus",font=("",14,"bold")).pack(side="left")
        ctk.CTkButton(top,text="Actualiser la liste",width=120,height=28,font=("",10),fg_color="#27272a",
                      command=self._universe_refresh).pack(side="right",padx=4)
        self._universe_count_lbl=ctk.CTkLabel(top,text="",font=("",10),text_color="#71717a"); self._universe_count_lbl.pack(side="right")
        # Auto-fill toolbar (target + region + button + progress)
        autofill=ctk.CTkFrame(tab,fg_color="transparent"); autofill.grid(row=1,column=0,columnspan=2,sticky="ew",pady=(0,6))
        autofill.grid_columnconfigure(1,weight=1)
        ctk.CTkLabel(autofill,text="Cible:",font=("",10),text_color="#a1a1aa").grid(row=0,column=0,padx=(0,4),pady=2)
        default_target=500
        if _engine_cfg and getattr(_engine_cfg,"DEFAULT_DATA_SETTINGS",None):
            default_target=_engine_cfg.DEFAULT_DATA_SETTINGS.get("data_min_tickers",500) or 500
        self._universe_target_entry=ctk.CTkEntry(autofill,width=80,height=28,font=("",10),placeholder_text="500")
        self._universe_target_entry.insert(0,str(default_target)); self._universe_target_entry.grid(row=0,column=1,padx=(0,8),pady=2,sticky="w")
        ctk.CTkLabel(autofill,text="Région:",font=("",10),text_color="#a1a1aa").grid(row=0,column=2,padx=(8,4),pady=2)
        self._universe_region_var=ctk.StringVar(value="global")
        self._universe_region_combo=ctk.CTkComboBox(autofill,values=["Global","US uniquement","Europe uniquement","Asie uniquement"],variable=self._universe_region_var,width=140,height=28,font=("",10))
        self._universe_region_combo.grid(row=0,column=3,padx=(0,8),pady=2)
        self._btn_auto_fill=ctk.CTkButton(autofill,text="Remplir automatiquement",width=180,height=28,font=("",10),fg_color="#4f46e5",command=self._on_auto_fill_clicked)
        self._btn_auto_fill.grid(row=0,column=4,padx=4,pady=2)
        self._universe_progress=ctk.CTkProgressBar(autofill,width=200,height=8); self._universe_progress.grid(row=0,column=5,padx=8,pady=2); self._universe_progress.grid_remove()
        self._universe_status_lbl=ctk.CTkLabel(autofill,text="",font=("",9),text_color="#71717a"); self._universe_status_lbl.grid(row=0,column=6,padx=4,pady=2); self._universe_status_lbl.grid_remove()
        self._btn_auto_fill_cancel=ctk.CTkButton(autofill,text="Annuler",width=80,height=28,font=("",10),fg_color="#7f1d1d",command=self._on_auto_fill_cancel); self._btn_auto_fill_cancel.grid(row=0,column=7,padx=4,pady=2); self._btn_auto_fill_cancel.grid_remove()
        self._universe_auto_fill_cancel_event=None
        # Left: list of tickers (Treeview) + add ISIN manually
        left=ctk.CTkFrame(tab,fg_color="#09090b",corner_radius=8,width=220); left.grid(row=2,column=0,sticky="nsew",padx=(0,4),pady=0)
        left.grid_propagate(False)
        left.grid_rowconfigure(1,weight=1); left.grid_columnconfigure(0,weight=1)
        ctk.CTkLabel(left,text="ISIN / Ticker",font=("",10,"bold"),text_color="#a1a1aa").grid(row=0,column=0,sticky="ew",padx=6,pady=4)
        self._universe_tree=None
        self._universe_tree, self._universe_sb = self._universe_build_list(left)
        self._universe_tree.grid(row=1,column=0,sticky="nsew",padx=4,pady=(0,4))
        if self._universe_sb: self._universe_sb.grid(row=1,column=1,sticky="ns",pady=(0,4))
        add_isin_f=ctk.CTkFrame(left,fg_color="transparent"); add_isin_f.grid(row=2,column=0,columnspan=2,sticky="ew",padx=6,pady=(4,6))
        add_isin_f.grid_columnconfigure(1,weight=1)
        ctk.CTkLabel(add_isin_f,text="ISIN:",font=("",10),text_color="#a1a1aa").grid(row=0,column=0,padx=(0,4),pady=2,sticky="w")
        self._universe_add_isin_entry=ctk.CTkEntry(add_isin_f,width=140,height=26,font=("",10),placeholder_text="ex. US0378331005")
        self._universe_add_isin_entry.grid(row=0,column=1,padx=(0,4),pady=2,sticky="ew")
        ctk.CTkButton(add_isin_f,text="Ajouter",width=70,height=26,font=("",10),fg_color="#27272a",command=self._universe_add_isin_clicked).grid(row=0,column=2,pady=2)
        # Right: chart area
        self._universe_chart=ctk.CTkFrame(tab,fg_color="#09090b",corner_radius=8); self._universe_chart.grid(row=2,column=1,sticky="nsew",padx=4,pady=0)
        self._universe_chart.grid_columnconfigure(0,weight=1); self._universe_chart.grid_rowconfigure(0,weight=1)
        ctk.CTkLabel(self._universe_chart,text="Cliquez sur un titre (ISIN/Ticker) pour afficher l'historique des prix",
                     text_color="#52525b",font=("",11)).pack(pady=40)
        self._universe_fill_tickers()

    def _universe_build_list(self,parent):
        t=ttk.Treeview(parent,columns=("isin","ticker"),show="headings",height=24,selectmode="browse",style="T.Treeview")
        t.heading("isin",text="ISIN"); t.column("isin",width=100,minwidth=80)
        t.heading("ticker",text="Ticker"); t.column("ticker",width=80,minwidth=60)
        t.bind("<<TreeviewSelect>>",self._universe_on_select)
        sb=ttk.Scrollbar(parent,orient="vertical",command=t.yview)
        t.configure(yscrollcommand=sb.set)
        return t, sb

    def _universe_fill_tickers(self):
        try:
            # Univers principal: table 'universe' dans le portfolio DB
            from core import portfolio
            universe = portfolio.get_universe() or {}
            tickers = sorted(universe.keys())
            imap = get_isin_map()
            n_isins = sum(1 for t in tickers if imap.get(t))
            if tickers:
                self._universe_count_lbl.configure(text=f"{len(tickers)} titres — {n_isins} ISIN")
            else:
                self._universe_count_lbl.configure(text="Aucun titre (lancer un scan ou un run modèle)")
        except Exception:
            tickers = []
            imap = {}
            self._universe_count_lbl.configure(text="")
        for c in self._universe_tree.get_children(""):
            self._universe_tree.delete(c)
        for sym in tickers:
            self._universe_tree.insert("", "end", values=(get_display_id(sym, imap), sym))

    def _universe_refresh(self):
        self._universe_fill_tickers()

    def _universe_add_isin_clicked(self):
        isin=(self._universe_add_isin_entry.get() or "").strip().upper().replace(" ","")
        if len(isin)<10:
            messagebox.showwarning("ISIN","Veuillez saisir un ISIN valide (12 caractères).")
            return
        try:
            from data.isin_mapper import map_isin_to_ticker
            tickers=list(get_stored_universe_list())
            imap=get_isin_map()
            ticker=map_isin_to_ticker(isin)
            key=ticker if ticker else isin
            if key in tickers and imap.get(key)==isin:
                messagebox.showinfo("ISIN","Cet ISIN est déjà dans l'univers.")
                return
            if key not in tickers:
                tickers.append(key)
            set_stored_universe_list(tickers)
            set_isin_map({key:isin})
            self._universe_add_isin_entry.delete(0,"end")
            self._universe_fill_tickers()
            messagebox.showinfo("ISIN",f"Ajouté: {key} — {isin}")
        except Exception as e:
            messagebox.showerror("ISIN",f"Erreur: {e}")

    def _on_auto_fill_clicked(self):
        # Ce bouton utilise désormais le nouveau scanner FMP + DB universe
        # 1) S'assurer que la clé FMP est disponible (comme pour le modèle)
        try:
            from core import portfolio
            fmp_key = getattr(portfolio, "get_setting", None) and portfolio.get_setting("fmp_key")
            if fmp_key:
                os.environ["FMP_API_KEY"] = fmp_key
                logger.info("_on_auto_fill_clicked: FMP_API_KEY loaded from settings")
            else:
                logger.warning("_on_auto_fill_clicked: no FMP key in settings")
        except Exception as e:
            logger.warning("_on_auto_fill_clicked: failed to load FMP key: %s", e)
        # 2) Lire la taille actuelle de l'univers (avant scan)
        try:
            from core import portfolio as _pf
            before_universe = _pf.get_universe() or {}
            before_n = len(before_universe)
            logger.info("_on_auto_fill_clicked: universe size before scan = %d", before_n)
        except Exception as e:
            logger.warning("_on_auto_fill_clicked: failed to load universe before scan: %s", e)
            before_n = 0
        # UI: état "en cours"
        self._btn_auto_fill.configure(state="disabled")
        self._universe_progress.set(0)
        self._universe_progress.grid()
        self._universe_status_lbl.grid()
        self._universe_status_lbl.configure(text="Scan de l'univers en cours…")
        self._btn_auto_fill_cancel.grid_remove()
        self._universe_auto_fill_cancel_event = None

        def scan_cb(msg):
            def _update():
                self._universe_status_lbl.configure(text=str(msg)[:85])
            self.after(0, _update)
            try:
                logger.info("Auto-fill scan status: %s", msg)
            except Exception:
                pass

        def worker():
            try:
                from core import data as core_data
                logger.info("_on_auto_fill_clicked: starting scan_and_expand_universe() worker")
                universe = core_data.scan_and_expand_universe(callback=scan_cb) or {}
                after_n = len(universe)
                added_n = max(0, after_n - before_n)
                logger.info(
                    "_on_auto_fill_clicked: scan done. before=%d after=%d added=%d",
                    before_n,
                    after_n,
                    added_n,
                )
                result = {
                    "added": list(universe.keys())[-added_n:] if added_n > 0 else [],
                    "failed": [],
                    "universe_size": after_n,
                    "target_reached": added_n > 0,
                    "sources_used": ["FMP screener"],
                }
            except Exception as e:
                err = str(e)
                logger.warning("_on_auto_fill_clicked: scan failed: %s", err)
                def _err():
                    messagebox.showerror("Auto-Fill", f"Erreur pendant le scan: {err}")
                self.after(0, _err)
                result = {
                    "added": [],
                    "failed": [],
                    "universe_size": before_n,
                    "target_reached": False,
                    "sources_used": [],
                }
            self.after(0, lambda r=result: self._on_auto_fill_complete(r))

        threading.Thread(target=worker, daemon=True).start()

    def _on_auto_fill_cancel(self):
        if getattr(self,"_universe_auto_fill_cancel_event",None):
            self._universe_auto_fill_cancel_event.set()

    def _on_auto_fill_complete(self,result):
        self._btn_auto_fill.configure(state="normal")
        self._universe_progress.grid_remove(); self._universe_status_lbl.grid_remove(); self._btn_auto_fill_cancel.grid_remove()
        self._universe_fill_tickers()
        target_str=self._universe_target_entry.get().strip() or "500"
        try: target=int(target_str)
        except ValueError: target=500
        added=len(result.get("added",[])); failed=len(result.get("failed",[])); size=result.get("universe_size",0); reached=result.get("target_reached",False)
        sources_used=result.get("sources_used",[])
        sources_str=", ".join(sources_used) if sources_used else "aucune"
        msg=f"Auto-fill terminé:\n  - Ajoutés: {added} actions\n  - Échec ISIN: {failed} (ignorés)\n  - Taille univers: {size}/{target}\n  - Sources: {sources_str}"
        if reached: msg+="\n  ✓ Cible atteinte"
        else: msg+="\n  ⚠ Cible non atteinte — toutes les sources ont été épuisées"
        messagebox.showinfo("Auto-Fill",msg)

    def _universe_on_select(self,ev):
        sel=self._universe_tree.selection()
        if not sel: return
        item=self._universe_tree.item(sel[0]); v=item.get("values")
        # v = (isin, ticker); yfinance uses ticker
        ticker=v[1] if v and len(v)>1 else (v[0] if v else None)
        if not ticker: return
        for w in self._universe_chart.winfo_children(): w.destroy()
        ctk.CTkLabel(self._universe_chart,text=f"Chargement de {ticker}…",text_color="#71717a",font=("",11)).pack(pady=30)
        def _load():
            try:
                import yfinance as yf
                from datetime import datetime, timedelta
                end=datetime.now(); start=end-timedelta(days=365*5)
                hist=yf.Ticker(ticker).history(start=start,end=end,auto_adjust=True)
                if hist is None or hist.empty:
                    self.after(0,lambda:self._universe_show_placeholder(ticker,"Pas de données"))
                    return
                self.after(0,lambda: self._universe_plot(ticker, hist))
            except Exception as e:
                self.after(0,lambda:self._universe_show_placeholder(ticker,str(e)))
        threading.Thread(target=_load,daemon=True).start()

    def _universe_show_placeholder(self,ticker,msg):
        for w in self._universe_chart.winfo_children(): w.destroy()
        ctk.CTkLabel(self._universe_chart,text=f"{ticker}: {msg}",text_color="#f87171",font=("",11)).pack(pady=30)

    def _universe_plot(self,ticker,hist):
        for w in self._universe_chart.winfo_children(): w.destroy()
        try:
            import matplotlib; matplotlib.use("Agg")
            from matplotlib.figure import Figure
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
            import numpy as np
            fig=Figure(figsize=(10,4),dpi=100,facecolor="#09090b")
            ax=fig.add_subplot(111); ax.set_facecolor("#09090b")
            close=hist["Close"] if "Close" in hist.columns else hist.iloc[:,0]
            ax.plot(close.index,close.values,color="#818cf8",linewidth=1.5)
            ax.fill_between(close.index,close.values,alpha=0.15,color="#818cf8")
            ax.set_title(f"{ticker} — Historique (5 ans)",fontsize=12,color="#e4e4e7")
            ax.tick_params(colors="#71717a",labelsize=9)
            for s in ["top","right"]: ax.spines[s].set_visible(False)
            for s in ["bottom","left"]: ax.spines[s].set_color("#27272a")
            ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x,p: f"{x:.0f}" if x>=1 else f"{x:.2f}"))
            fig.tight_layout(pad=1)
            cv=FigureCanvasTkAgg(fig,master=self._universe_chart); cv.draw(); cv.get_tk_widget().pack(fill="both",expand=True)
        except Exception as e:
            ctk.CTkLabel(self._universe_chart,text=f"Erreur: {e}",text_color="#f87171",font=("",11)).pack(pady=30)

    # ── SETTINGS TAB ──────────────────────────────────────────
    def _init_settings(self):
        tab=self.tabs.tab("Settings"); tab.grid_columnconfigure(1,weight=1)
        # Use a scrollable frame for all settings
        scroll=ctk.CTkScrollableFrame(tab,fg_color="transparent")
        scroll.pack(fill="both",expand=True,padx=5,pady=5)
        scroll.grid_columnconfigure(1,weight=1)
        self._sett={}
        self._engine_vars = {}
        self._engine_widgets = {}
        self._risk_widgets = {}
        row=0

        # Data freshness (visible last update timestamps)
        ctk.CTkLabel(scroll,text="Data freshness",font=("",15,"bold")).grid(row=row,column=0,columnspan=3,sticky="w",pady=(5,8)); row+=1
        info=getattr(self,"model_info",None) or {}
        fresh=info.get("data_freshness") or {}
        def _disp(dataset):
            d=fresh.get(dataset) if isinstance(fresh,dict) else {}
            return (d.get("display") or d.get("last_update_timestamp","—")[:16]) if isinstance(d,dict) else "—"
        ctk.CTkLabel(scroll,text="Prices last updated:",font=("",11)).grid(row=row,column=0,sticky="w",padx=8,pady=3)
        ctk.CTkLabel(scroll,text=_disp("prices"),font=("JetBrains Mono",10),text_color="#34d399").grid(row=row,column=1,sticky="w",pady=3); row+=1
        ctk.CTkLabel(scroll,text="Fundamentals last updated:",font=("",11)).grid(row=row,column=0,sticky="w",padx=8,pady=3)
        ctk.CTkLabel(scroll,text=_disp("fundamentals"),font=("JetBrains Mono",10),text_color="#34d399").grid(row=row,column=1,sticky="w",pady=3); row+=1
        ctk.CTkLabel(scroll,text="Macro last updated:",font=("",11)).grid(row=row,column=0,sticky="w",padx=8,pady=3)
        ctk.CTkLabel(scroll,text=_disp("macro"),font=("JetBrains Mono",10),text_color="#34d399").grid(row=row,column=1,sticky="w",pady=3); row+=1
        if not fresh and info.get("saved_at"):
            ctk.CTkLabel(scroll,text="Cache saved:",font=("",11)).grid(row=row,column=0,sticky="w",padx=8,pady=3)
            try: t=__import__("datetime").datetime.fromisoformat(str(info["saved_at"]).replace("Z","+00:00")).strftime("%Y-%m-%d %H:%M")
            except: t=str(info.get("saved_at",""))[:16]
            ctk.CTkLabel(scroll,text=t,font=("JetBrains Mono",10),text_color="#34d399").grid(row=row,column=1,sticky="w",pady=3); row+=1
        row+=1

        # API Keys
        ctk.CTkLabel(scroll,text="API Keys",font=("",15,"bold")).grid(row=row,column=0,columnspan=3,sticky="w",pady=(5,8)); row+=1
        for k,l,h in [("anthropic_key","Anthropic","console.anthropic.com"),
            ("fred_key","FRED","fred.stlouisfed.org"),("fmp_key","FMP","financialmodelingprep.com (requis pour modèle complet)"),
            ("av_key","Alpha Vantage","alphavantage.co (optionnel, fallback prix)")]:
            ctk.CTkLabel(scroll,text=l,font=("",11)).grid(row=row,column=0,sticky="w",padx=8,pady=3)
            e=ctk.CTkEntry(scroll,width=300,font=("JetBrains Mono",10),show="*")
            e.grid(row=row,column=1,sticky="w",pady=3); e.insert(0,portfolio.get_setting(k,""))
            self._sett[k]=e
            ctk.CTkLabel(scroll,text=h,font=("",9),text_color="#52525b").grid(row=row,column=2,sticky="w",padx=8)
            row+=1

        # Portfolio decision engine (transaction costs, horizon)
        ctk.CTkLabel(scroll,text="Portfolio decision engine",font=("",15,"bold")).grid(row=row,column=0,columnspan=3,sticky="w",pady=(15,8)); row+=1
        for k,l,d,hint in [
            ("tx_cost_broker_fee","Broker fee (EUR/trade)","0","Fixed fee per trade"),
            ("tx_cost_spread_bps","Spread (bps)","10","Bid-ask spread in basis points"),
            ("tx_cost_slippage_bps","Slippage (bps)","5","Slippage in bps"),
            ("default_holding_horizon_days","Default holding horizon (days)","365","Exit after this many days if set"),
        ]:
            ctk.CTkLabel(scroll,text=l,font=("",11)).grid(row=row,column=0,sticky="w",padx=8,pady=3)
            e=ctk.CTkEntry(scroll,width=200,font=("JetBrains Mono",10))
            e.grid(row=row,column=1,sticky="w",pady=3); e.insert(0,portfolio.get_setting(k,d))
            self._sett[k]=e
            ctk.CTkLabel(scroll,text=hint,font=("",9),text_color="#52525b").grid(row=row,column=2,sticky="w",padx=8)
            row+=1
        # Investment horizon (per strategy type, max holding, review frequency)
        if _engine_cfg:
            ps = _engine_cfg.get_portfolio_settings()
            ctk.CTkLabel(scroll,text="Investment horizon (per strategy)",font=("",15,"bold")).grid(row=row,column=0,columnspan=3,sticky="w",pady=(12,8)); row+=1
            for k,l,d,hint in [
                ("horizon_short_term_days","SHORT_TERM horizon (days)","60","30-90 typical"),
                ("horizon_medium_term_days","MEDIUM_TERM horizon (days)","180","90-365 typical"),
                ("horizon_long_term_days","LONG_TERM horizon (days)","365","Multi-year"),
                ("max_holding_duration_days","Max holding duration (days)","730","Cap exit by date"),
                ("review_frequency_days","Review frequency (days)","30","Next review date step"),
            ]:
                ctk.CTkLabel(scroll,text=l,font=("",11)).grid(row=row,column=0,sticky="w",padx=8,pady=3)
                e=ctk.CTkEntry(scroll,width=120,font=("JetBrains Mono",10))
                e.grid(row=row,column=1,sticky="w",pady=3); e.insert(0,str(ps.get(k,d)))
                self._sett[k]=e
                ctk.CTkLabel(scroll,text=hint,font=("",9),text_color="#52525b").grid(row=row,column=2,sticky="w",padx=8)
                row+=1
            # Minimum expected alpha: net return after costs must exceed this (stored as fraction; UI shows %)
            ctk.CTkLabel(scroll,text="Minimum expected alpha (%)",font=("",11)).grid(row=row,column=0,sticky="w",padx=8,pady=3)
            e_min_alpha=ctk.CTkEntry(scroll,width=120,font=("JetBrains Mono",10))
            e_min_alpha.grid(row=row,column=1,sticky="w",pady=3)
            e_min_alpha.insert(0, str(round(ps.get("minimum_expected_alpha", 0.01) * 100, 1)))
            self._sett["minimum_expected_alpha_pct"]=e_min_alpha
            ctk.CTkLabel(scroll,text="Min net return after costs (e.g. 1 = 1%)",font=("",9),text_color="#52525b").grid(row=row,column=2,sticky="w",padx=8)
            row+=1
            # Sell signal configuration (per strategy)
            sell_modes = ["disabled", "passive", "active"]
            ctk.CTkLabel(scroll, text="Sell Signal Configuration", font=("", 15, "bold")).grid(row=row, column=0, columnspan=3, sticky="w", pady=(12, 8)); row += 1
            ctk.CTkLabel(scroll, text="Disabled = never sell; Passive = stop-loss only; Active = all signals.", font=("", 9), text_color="#52525b").grid(row=row, column=0, columnspan=3, sticky="w", padx=8); row += 1
            self._sell_mode_vars = {}
            for k, label in [
                ("sell_mode_long_term", "Long Term"),
                ("sell_mode_medium_term", "Medium Term"),
                ("sell_mode_short_term", "Short Term"),
                ("sell_mode_speculative", "Speculative"),
            ]:
                ctk.CTkLabel(scroll, text=label, font=("", 11)).grid(row=row, column=0, sticky="w", padx=8, pady=3)
                var = tk.StringVar(value=ps.get(k, "disabled" if "long" in k else "passive" if "medium" in k else "active"))
                ctk.CTkOptionMenu(scroll, values=sell_modes, variable=var, width=180).grid(row=row, column=1, sticky="w", pady=3)
                self._sell_mode_vars[k] = var
                row += 1
            # Risk & regime (configurable limits and regime detection thresholds)
            rs = _engine_cfg.get_risk_settings()
            ctk.CTkLabel(scroll,text="Risk & regime",font=("",15,"bold")).grid(row=row,column=0,columnspan=3,sticky="w",pady=(15,8)); row+=1
            for k,l,d,hint in [
                ("max_sector_weight","Max sector weight (0-1)","0.35","Cap per sector"),
                ("max_turnover_pct","Max turnover (0-1)","0.5","Limit trading"),
                ("max_position_pct","Max single position (0-1)","0.15","Cap per asset"),
                ("bull_return_6m","Bull threshold (6m return)","0.05","Above = bull"),
                ("bear_return_6m","Bear threshold (6m return)","-0.05","Below = bear"),
                ("high_vol_threshold","High vol threshold","0.25","Annualized vol"),
                ("low_vol_threshold","Low vol threshold","0.15","Annualized vol"),
            ]:
                ctk.CTkLabel(scroll,text=l,font=("",11)).grid(row=row,column=0,sticky="w",padx=8,pady=3)
                e=ctk.CTkEntry(scroll,width=100,font=("JetBrains Mono",10))
                e.grid(row=row,column=1,sticky="w",pady=3); e.insert(0,str(rs.get(k,d)))
                self._risk_widgets[k]=e
                ctk.CTkLabel(scroll,text=hint,font=("",9),text_color="#52525b").grid(row=row,column=2,sticky="w",padx=8)
                row+=1

        # Engine / Model configuration (DB-backed)
        if _engine_cfg:
            try:
                _engine_cfg.init_default_config()
            except Exception:
                pass
            ctk.CTkLabel(scroll,text="Engine / Model configuration",font=("",15,"bold")).grid(row=row,column=0,columnspan=3,sticky="w",pady=(15,8)); row+=1
            mset = _engine_cfg.get_model_settings()
            fset = _engine_cfg.get_feature_settings()
            enabled = mset.get("enabled_models") or _engine_cfg.MODEL_IDS
            for mid in _engine_cfg.MODEL_IDS:
                v = tk.BooleanVar(value=mid in enabled)
                ctk.CTkCheckBox(scroll,text=mid,variable=v,font=("",11)).grid(row=row,column=0,columnspan=2,sticky="w",padx=8,pady=2)
                self._engine_vars["model_"+mid]=v
                row+=1
            ctk.CTkLabel(scroll,text="Execution mode",font=("",11)).grid(row=row,column=0,sticky="w",padx=8,pady=3)
            em_var = tk.StringVar(value=mset.get("execution_mode","all"))
            em_dd = ctk.CTkOptionMenu(scroll,values=_engine_cfg.EXECUTION_MODES,variable=em_var,width=120)
            em_dd.grid(row=row,column=1,sticky="w",pady=3); self._engine_vars["execution_mode"]=em_var; row+=1
            ctk.CTkLabel(scroll,text="Single model (when mode=single)",font=("",11)).grid(row=row,column=0,sticky="w",padx=8,pady=3)
            sm_var = tk.StringVar(value=mset.get("single_model_id","LightGBM"))
            ctk.CTkOptionMenu(scroll,values=_engine_cfg.MODEL_IDS,variable=sm_var,width=120).grid(row=row,column=1,sticky="w",pady=3)
            self._engine_vars["single_model_id"]=sm_var; row+=1
            ctk.CTkLabel(scroll,text="Ensemble method",font=("",11)).grid(row=row,column=0,sticky="w",padx=8,pady=3)
            ens_var = tk.StringVar(value=mset.get("ensemble_method","ic_weighted_average"))
            ctk.CTkOptionMenu(scroll,values=_engine_cfg.ENSEMBLE_METHODS,variable=ens_var,width=180).grid(row=row,column=1,sticky="w",pady=3)
            self._engine_vars["ensemble_method"]=ens_var; row+=1
            ctk.CTkLabel(scroll,text="Prediction horizon (months)",font=("",11)).grid(row=row,column=0,sticky="w",padx=8,pady=3)
            hz_var = tk.StringVar(value=str(mset.get("prediction_horizon_months",12)))
            ctk.CTkOptionMenu(scroll,values=["3","6","12","24","120"],variable=hz_var,width=80).grid(row=row,column=1,sticky="w",pady=3)
            self._engine_vars["prediction_horizon_months"]=hz_var; row+=1
            for k,l in [("winsorization","Winsorization"),("rank_normalization","Rank normalization"),("sector_neutralization","Sector neutralization"),("feature_decorrelation","Feature decorrelation (PCA)"),("use_gpu","Use GPU for TCN/LSTM (auto-detect)")]:
                v = tk.BooleanVar(value=mset.get(k,True if k!="feature_decorrelation" else False))
                ctk.CTkCheckBox(scroll,text=l,variable=v,font=("",11)).grid(row=row,column=0,columnspan=2,sticky="w",padx=8,pady=2)
                self._engine_vars[k]=v; row+=1
            ctk.CTkLabel(scroll,text="PCA variance ratio (if decorrelation on)",font=("",11)).grid(row=row,column=0,sticky="w",padx=8,pady=3)
            pca_var=ctk.CTkEntry(scroll,width=80,font=("JetBrains Mono",10)); pca_var.insert(0,str(mset.get("pca_variance_ratio",0.95)))
            pca_var.grid(row=row,column=1,sticky="w",pady=3); self._engine_widgets["pca_variance_ratio"]=pca_var; row+=1
            ctk.CTkLabel(scroll,text="Term structure weights (3m, 6m, 12m, 24m)",font=("",11)).grid(row=row,column=0,sticky="w",padx=8,pady=3); row+=1
            for h,l in [("3m","3M"),("6m","6M"),("12m","12M"),("24m","24M")]:
                ctk.CTkLabel(scroll,text=l,font=("",10)).grid(row=row,column=0,sticky="w",padx=20,pady=1)
                e=ctk.CTkEntry(scroll,width=60,font=("JetBrains Mono",10)); e.insert(0,str(mset.get("horizon_weight_"+h,0.5 if h=="12m" else 0.1)))
                e.grid(row=row,column=1,sticky="w",pady=1); self._engine_widgets["horizon_weight_"+h]=e; row+=1
            for k,l,d in [("lookback_days","Lookback (days)",252),("training_window_years","Training window (years)",3),("retraining_frequency_months","Retraining (months)",3),
                ("n_estimators","n_estimators",500),("max_depth","max_depth",5),("learning_rate","Learning rate",0.03),("ridge_alpha","Ridge alpha",10.0),
                ("winsorize_quantile","Winsorize quantile",0.02)]:
                ctk.CTkLabel(scroll,text=l,font=("",11)).grid(row=row,column=0,sticky="w",padx=8,pady=3)
                e=ctk.CTkEntry(scroll,width=100,font=("JetBrains Mono",10))
                e.grid(row=row,column=1,sticky="w",pady=3); e.insert(0,str(mset.get(k,d)))
                self._engine_widgets[k]=e; row+=1
            ctk.CTkLabel(scroll,text="Feature groups",font=("",13,"bold")).grid(row=row,column=0,columnspan=3,sticky="w",pady=(10,6)); row+=1
            for gname in _engine_cfg.FEATURE_GROUPS:
                v = tk.BooleanVar(value=fset.get(gname,True))
                ctk.CTkCheckBox(scroll,text=gname,variable=v,font=("",11)).grid(row=row,column=0,columnspan=2,sticky="w",padx=8,pady=2)
                self._engine_vars["feat_"+gname]=v; row+=1

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

        btn_row=ctk.CTkFrame(scroll,fg_color="transparent"); btn_row.grid(row=row,column=0,columnspan=3,sticky="w",pady=15,padx=8); row+=1
        ctk.CTkButton(btn_row,text="Apply",width=120,height=36,font=("",12,"bold"),
                      fg_color="#4f46e5",hover_color="#4338ca",command=self._apply_settings).pack(side="left",padx=(0,8))
        ctk.CTkButton(btn_row,text="Save All Settings",width=180,height=36,font=("",12,"bold"),
                      fg_color="#047857",command=self._save_settings).pack(side="left")

    def _apply_settings(self):
        """Save all settings, invalidate model cache, and refresh so new config applies on next Run/Build. No restart."""
        self._save_settings()
        model.clear_model_cache()
        self.model_results=None
        self.feat_imp=None
        self.model_info=None
        if hasattr(self,"rk_tree") and self.rk_tree.winfo_exists():
            for c in self.rk_tree.get_children(): self.rk_tree.delete(c)
        if hasattr(self,"bld_tree") and self.bld_tree.winfo_exists():
            for c in self.bld_tree.get_children(): self.bld_tree.delete(c)
        self._build_proposals=[]
        messagebox.showinfo("Apply","Settings applied. Click \"Run Model\" (Rankings) to recalculate with new parameters, then \"Generate\" (Build) to refresh recommendations.")

    def _save_settings(self):
        for k,e in self._sett.items(): portfolio.set_setting(k,e.get())
        for k,ev in [("fred_key","FRED_API_KEY"),("fmp_key","FMP_API_KEY"),("av_key","ALPHA_VANTAGE_KEY")]:
            v=portfolio.get_setting(k)
            if v: os.environ[ev]=v
        if _engine_cfg and getattr(self,"_engine_vars",None) and getattr(self,"_engine_widgets",None):
            try:
                def _ev(k, default=False):
                    v = self._engine_vars.get(k)
                    return v.get() if v is not None else default
                def _ew(k, default, cast=int):
                    w = self._engine_widgets.get(k)
                    raw = w.get() if w else ""
                    if not raw: return default
                    try: return cast(raw)
                    except: return default
                enabled = [mid for mid in _engine_cfg.MODEL_IDS if _ev("model_"+mid)]
                vo = self._engine_vars.get("execution_mode")
                sm = self._engine_vars.get("single_model_id")
                em = self._engine_vars.get("ensemble_method")
                hz_var = self._engine_vars.get("prediction_horizon_months")
                hz_val = int(hz_var.get()) if hz_var else 12
                mset = {
                    "enabled_models": enabled or _engine_cfg.MODEL_IDS[:4],
                    "execution_mode": vo.get() if vo else "all",
                    "single_model_id": sm.get() if sm else "LightGBM",
                    "ensemble_method": em.get() if em else "ic_weighted_average",
                    "prediction_horizon_months": hz_val,
                    "winsorization": _ev("winsorization", True),
                    "rank_normalization": _ev("rank_normalization", True),
                    "sector_neutralization": _ev("sector_neutralization", True),
                    "use_gpu": _ev("use_gpu", True),
                    "lookback_days": _ew("lookback_days", 252),
                    "training_window_years": _ew("training_window_years", 3),
                    "retraining_frequency_months": _ew("retraining_frequency_months", 3),
                    "n_estimators": _ew("n_estimators", 500),
                    "max_depth": _ew("max_depth", 5),
                    "learning_rate": _ew("learning_rate", 0.03, float),
                    "ridge_alpha": _ew("ridge_alpha", 10.0, float),
                    "winsorize_quantile": _ew("winsorize_quantile", 0.02, float),
                    "feature_decorrelation": _ev("feature_decorrelation", False),
                }
                if "pca_variance_ratio" in self._engine_widgets:
                    mset["pca_variance_ratio"] = _ew("pca_variance_ratio", 0.95, float)
                for h in ["3m","6m","12m","24m"]:
                    if "horizon_weight_"+h in self._engine_widgets:
                        mset["horizon_weight_"+h] = _ew("horizon_weight_"+h, 0.5 if h=="12m" else 0.1, float)
                _engine_cfg.set_system_config(_engine_cfg.DOMAIN_MODEL, mset)
                fset = {gname: _ev("feat_"+gname, True) for gname in _engine_cfg.FEATURE_GROUPS}
                _engine_cfg.set_system_config(_engine_cfg.DOMAIN_FEATURE, fset)
                risk = _engine_cfg.get_risk_settings()
                for k, w in getattr(self, "_risk_widgets", {}).items():
                    if hasattr(w, "get"):
                        try: risk[k] = float(w.get())
                        except ValueError as e:
                            logger.debug("Settings risk widget %s: %s", k, e)
                _engine_cfg.set_system_config(_engine_cfg.DOMAIN_RISK, risk)
            except Exception as ex:
                messagebox.showwarning("Engine config", f"Engine settings may not have saved: {ex}")
            # Persist investment horizon (portfolio_settings)
            try:
                ps = _engine_cfg.get_portfolio_settings()
                for k in ("horizon_short_term_days", "horizon_medium_term_days", "horizon_long_term_days",
                          "max_holding_duration_days", "review_frequency_days"):
                    e = self._sett.get(k)
                    if e is not None:
                        raw = e.get()
                        if raw:
                            try: ps[k] = int(raw)
                            except ValueError as e:
                                logger.debug("Settings portfolio %s int: %s", k, e)
                e_min_alpha = self._sett.get("minimum_expected_alpha_pct")
                if e_min_alpha is not None:
                    try:
                        pct = float(e_min_alpha.get())
                        if 0 <= pct <= 100:
                            ps["minimum_expected_alpha"] = pct / 100.0
                    except ValueError as e:
                        logger.debug("Settings minimum_expected_alpha: %s", e)
                for k in getattr(self, "_sell_mode_vars", {}):
                    v = self._sell_mode_vars[k].get()
                    if v in ("disabled", "passive", "active"):
                        ps[k] = v
                ps["sell_mode_dont_sell"] = "disabled"
                _engine_cfg.set_system_config(_engine_cfg.DOMAIN_PORTFOLIO, ps)
            except Exception as e:
                logger.warning("Settings save portfolio/sell_mode: %s", e)
        self._update_engine()
        messagebox.showinfo("OK","Settings saved. Use \"Apply\" to invalidate cache and apply to next run.")

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
