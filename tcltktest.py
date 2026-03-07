try:
    import tkinter  # Python 3.x 用tkinter，Python 2.x 用Tkinter
    # 验证tkinter是否能正常初始化
    root = tkinter.Tk()
    root.withdraw()  # 隐藏窗口（避免弹出空白窗口）
    print("✅ Python已安装Tcl/Tk组件，版本信息：")
    print(f"tkinter版本: {tkinter.TkVersion}")
    print(f"Tcl版本: {tkinter.TclVersion}")
except ImportError:
    print("❌ Python未安装Tcl/Tk组件（缺少tkinter模块）")
except Exception as e:
    print(f"❌ Tcl/Tk组件存在但无法使用，错误信息：{e}")
    