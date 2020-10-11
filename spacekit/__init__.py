def reload(mod):
    """Reloads the module from file without restarting kernel.
        Args:
            mod (loaded mod or list of mod objects): name or handle of package (i.e.,[ pd, fs,np])
        Returns:
            reload each model.
    Example:
    # You pass in whatever name you imported as.
    import my_functions_from_file as mf
    # after editing the source file:
    # mf.reload(mf)"""
    from importlib import reload
    print(f'Reloading...\n')
    
    ## 
    if isinstance(mod,list):
        return [reload(m) for m in mod]
    else:
        return  reload(mod)
    
