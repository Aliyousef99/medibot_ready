import React, { useEffect, useState } from 'react';
import {
  getRecommendationsForLab,
  RecommendationSet,
  getHistory,
  getHistoryDetail,
  deleteHistory,
  LabReport,
} from '../services/api';
import { Eye, Trash2, X, Loader2, AlertCircle, CheckCircle2, TriangleAlert, Info, RefreshCw } from 'lucide-react';
import { useToastStore } from '../state/toastStore';
import { handleApiError } from '../services/api';

const HistoryPage: React.FC = () => {
  const [labs, setLabs] = useState<LabReport[]>([]);
  const [selectedLab, setSelectedLab] = useState<LabReport | null>(null);
  const [labToDelete, setLabToDelete] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [detailLoading, setDetailLoading] = useState(false);
  const [deleting, setDeleting] = useState(false);
  const toast = useToastStore();

  const fetchHistory = async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await getHistory();
      setLabs(data);
    } catch (err: any) {
      handleApiError(err);
      setError(`Failed to fetch lab history: ${err.message || "Network error"}`);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchHistory();
  }, []);

  const handleViewDetails = async (labId: string) => {
    setDetailLoading(true);
    setError(null);
    try {
      const data = await getHistoryDetail(labId);
      setSelectedLab(data);
    } catch (err: any) {
      handleApiError(err);
      setError(`Failed to fetch lab details: ${err.message || "Network error"}`);
    } finally {
      setDetailLoading(false);
    }
  };

  const handleDelete = async () => {
    if (labToDelete === null) return;
    setDeleting(true);
    setError(null);
    try {
      await deleteHistory(labToDelete);
      setLabs(labs.filter((lab) => lab.id !== labToDelete));
      setLabToDelete(null); // Close confirmation dialog
    } catch (err: any) {
      handleApiError(err);
      setError(`Failed to delete lab report: ${err.message || "Network error"}`);
    } finally {
      setDeleting(false);
    }
  };

  if (loading) {
    return (
      <div className="p-6 flex items-center justify-center">
        <Loader2 className="w-8 h-8 animate-spin mr-2" />
        <span>Loading history...</span>
      </div>
    );
  }

  return (
    <div className="p-4 md:p-6">
      <h1 className="text-2xl font-bold mb-4">Lab Report History</h1>
      
      {error && (
        <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded relative mb-4" role="alert">
          <strong className="font-bold">Error: </strong>
          <span className="block sm:inline">{error}</span>
          <div className="absolute top-0 bottom-0 right-0 flex items-center gap-2 px-4 py-3">
            <button
              className="text-xs px-2 py-1 rounded border border-red-300 hover:bg-red-50"
              onClick={fetchHistory}
            >
              Retry
            </button>
            <button onClick={() => setError(null)} aria-label="Dismiss error">
              <X className="h-6 w-6 text-red-500" />
            </button>
          </div>
        </div>
      )}

      {labs.length === 0 && !loading ? (
        <div className="text-center py-12">
          <p className="text-gray-500">You have no saved lab reports yet.</p>
          <button
            onClick={fetchHistory}
            className="mt-3 inline-flex items-center gap-2 px-3 py-2 rounded-lg border border-zinc-300 hover:bg-zinc-100"
          >
            <RefreshCw className="w-4 h-4" /> Refresh
          </button>
        </div>
      ) : (
        <div className="space-y-4">
          {labs.map((lab) => (
            <div key={lab.id} className="p-4 border rounded-lg flex justify-between items-center">
              <div>
                <p className="font-semibold">Report from {new Date(lab.created_at).toLocaleDateString()}</p>
                <p className="text-sm text-gray-600 truncate max-w-md">{lab.summary}</p>
              </div>
              <div className="flex items-center space-x-2">
                <button onClick={() => handleViewDetails(lab.id)} className="p-2 hover:bg-gray-100 rounded disabled:opacity-50" disabled={detailLoading}>
                  {detailLoading && selectedLab?.id !== lab.id ? <Loader2 className="w-5 h-5 animate-spin"/> : <Eye size={20} />}
                </button>
                <button onClick={() => setLabToDelete(lab.id)} className="p-2 hover:bg-red-100 rounded disabled:opacity-50" disabled={deleting}>
                  <Trash2 size={20} className="text-red-500" />
                </button>
              </div>
            </div>
          ))}
        </div>
      )}

      {selectedLab && (
        <LabDetailModal lab={selectedLab} onClose={() => setSelectedLab(null)} />
      )}

      {labToDelete !== null && (
        <ConfirmationDialog
          onConfirm={handleDelete}
          onCancel={() => setLabToDelete(null)}
          loading={deleting}
        />
      )}
    </div>
  );
};

const LabDetailModal: React.FC<{ lab: LabReport; onClose: () => void }> = ({ lab, onClose }) => {
  const [recs, setRecs] = useState<RecommendationSet | null>(null);
  const [recError, setRecError] = useState<string | null>(null);
  const [recLoading, setRecLoading] = useState(false);

  async function handleGetRecs() {
    setRecError(null);
    setRecLoading(true);
    try {
      const r = await getRecommendationsForLab(lab.id);
      setRecs(r);
    } catch (e: any) {
      setRecError(e?.message || 'Failed to get recommendations');
    } finally {
      setRecLoading(false);
    }
  }

  function tierClass(t: string) {
    if (t === 'high') return 'bg-red-50 text-red-700 border-red-200';
    if (t === 'moderate') return 'bg-amber-50 text-amber-700 border-amber-200';
    return 'bg-emerald-50 text-emerald-700 border-emerald-200';
  }

  function tierIcon(t: string) {
    if (t === 'high') return <TriangleAlert className="w-4 h-4"/>;
    if (t === 'moderate') return <AlertCircle className="w-4 h-4"/>;
    return <CheckCircle2 className="w-4 h-4"/>;
  }

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex justify-center items-center p-4 z-50">
      <div className="bg-white rounded-lg shadow-xl p-6 w-full max-w-2xl max-h-full overflow-y-auto">
        <div className="flex justify-between items-center mb-4">
          <h2 className="text-xl font-bold">Lab Report Details</h2>
          <button onClick={onClose} className="p-2 hover:bg-gray-100 rounded-full">
            <X size={24} />
          </button>
        </div>
        <div className="space-y-4">
          <div>
            <h3 className="font-semibold">Date</h3>
            <p>{new Date(lab.created_at).toLocaleString()}</p>
          </div>
          <div>
            <h3 className="font-semibold">AI Summary</h3>
            <p className="text-gray-700 whitespace-pre-wrap">{lab.summary}</p>
          </div>
          <div>
            <h3 className="font-semibold">Structured Data</h3>
            <pre className="bg-gray-100 p-4 rounded-lg text-sm overflow-x-auto">
              {JSON.stringify(lab.structured_json, null, 2)}
            </pre>
          </div>
          <div className="pt-2">
            <button
              onClick={handleGetRecs}
              disabled={recLoading}
              className="inline-flex items-center gap-2 px-4 py-2 rounded-lg bg-emerald-600 text-white hover:bg-emerald-500 disabled:opacity-60"
            >
              {recLoading && <Loader2 className="w-4 h-4 animate-spin"/>}
              Get Recommendations
            </button>
            {recError && (
              <div className="mt-3 text-sm text-red-600">{recError}</div>
            )}
            {recs && (
              <div className={`mt-4 rounded-xl border p-4 ${tierClass(recs.risk_tier)}`}>
                <div className="flex items-center gap-2 font-semibold mb-2">
                  {tierIcon(recs.risk_tier)}
                  <span>Risk tier: {recs.risk_tier}</span>
                  {!recs.llm_used && (
                    <span className="ml-auto text-[10px] px-2 py-0.5 rounded bg-white/60 text-zinc-600 flex items-center gap-1">
                      <Info className="w-3 h-3"/> Generated by rules engine
                    </span>
                  )}
                </div>
                <ul className="space-y-1 text-sm">
                  {recs.actions.map((a, i) => (
                    <li key={i} className="flex items-start gap-2">
                      <CheckCircle2 className="w-4 h-4 mt-0.5"/>
                      <span>{a}</span>
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

const ConfirmationDialog: React.FC<{ onConfirm: () => void; onCancel: () => void; loading: boolean }> = ({ onConfirm, onCancel, loading }) => {
  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex justify-center items-center p-4 z-50">
      <div className="bg-white rounded-lg shadow-xl p-6 w-full max-w-sm">
        <h2 className="text-lg font-semibold mb-4">Are you sure?</h2>
        <p className="mb-6">Do you want to permanently delete this lab report?</p>
        <div className="flex justify-end space-x-4">
          <button onClick={onCancel} className="px-4 py-2 rounded-lg border hover:bg-gray-100" disabled={loading}>
            Cancel
          </button>
          <button onClick={onConfirm} className="px-4 py-2 rounded-lg bg-red-500 text-white hover:bg-red-600 flex items-center gap-2 disabled:opacity-50" disabled={loading}>
            {loading && <Loader2 className="w-4 h-4 animate-spin"/>}
            {loading ? 'Deleting...' : 'Delete'}
          </button>
        </div>
      </div>
    </div>
  );
};

export default HistoryPage;
