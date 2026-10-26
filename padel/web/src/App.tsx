import { useEffect, useState } from 'react'
import Calibrate from './pages/Calibrate'
import Matches from './pages/Matches'
import ReportPage from './pages/Report'

// Hash routes: #/  ·  #/m/<id>/calibrate  ·  #/m/<id>
function useRoute(): string[] {
  const read = () => window.location.hash.replace(/^#\/?/, '').split('/').filter(Boolean)
  const [parts, setParts] = useState(read)
  useEffect(() => {
    const on = () => setParts(read())
    window.addEventListener('hashchange', on)
    return () => window.removeEventListener('hashchange', on)
  }, [])
  return parts
}

export const go = (path: string) => {
  window.location.hash = path
}

export default function App() {
  const parts = useRoute()
  let page = <Matches />
  let crumb = ''
  if (parts[0] === 'm' && parts[1]) {
    if (parts[2] === 'calibrate') {
      page = <Calibrate id={parts[1]} />
      crumb = 'Court calibration'
    } else {
      page = <ReportPage id={parts[1]} />
      crumb = 'Match report'
    }
  }
  return (
    <>
      <header className="topbar">
        <a className="brand" href="#/">
          <img src="/favicon.svg" alt="" />
          PadelLens
        </a>
        {crumb && <span className="crumbs">/ {crumb}</span>}
      </header>
      <main className="wrap">{page}</main>
    </>
  )
}
