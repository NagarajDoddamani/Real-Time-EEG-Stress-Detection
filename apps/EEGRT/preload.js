const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('eegApp', {
  discoverStreams: () => ipcRenderer.invoke('lsl:discover'),
  startLive: details => ipcRenderer.invoke('live:start', details),
  stopLive: () => ipcRenderer.invoke('live:stop'),
  listSessions: limit => ipcRenderer.invoke('sessions:list', limit),
  getSessionReport: sessionId => ipcRenderer.invoke('session:report', sessionId),
  savePdfReport: report => ipcRenderer.invoke('report:pdf', report),
  onLiveEvent: callback => {
    const listener = (_event, payload) => callback(payload);
    ipcRenderer.on('live:event', listener);
    return () => ipcRenderer.removeListener('live:event', listener);
  },
  onLiveExit: callback => {
    const listener = (_event, payload) => callback(payload);
    ipcRenderer.on('live:exit', listener);
    return () => ipcRenderer.removeListener('live:exit', listener);
  }
});
