const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('eegApp', {
  listSessions: limit => ipcRenderer.invoke('sessions:list', limit),
  predictSession: sessionId => ipcRenderer.invoke('session:predict', sessionId),
  startMonitor: sessionId => ipcRenderer.invoke('monitor:start', sessionId),
  stopMonitor: () => ipcRenderer.invoke('monitor:stop'),
  savePdfReport: report => ipcRenderer.invoke('report:pdf', report),
  onMonitorEvent: callback => {
    const listener = (_event, payload) => callback(payload);
    ipcRenderer.on('monitor:event', listener);
    return () => ipcRenderer.removeListener('monitor:event', listener);
  },
  onMonitorExit: callback => {
    const listener = (_event, payload) => callback(payload);
    ipcRenderer.on('monitor:exit', listener);
    return () => ipcRenderer.removeListener('monitor:exit', listener);
  }
});
