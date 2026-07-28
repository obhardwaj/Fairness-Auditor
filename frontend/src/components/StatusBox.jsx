export function LoadingSkeleton({ lines = 3 }) {
    return (
      <div className="animate-pulse space-y-2">
        {Array.from({ length: lines }).map((_, i) => (
          <div
            key={i}
            className="h-4 rounded bg-gray-200"
            style={{ width: `${85 - i * 15}%` }}
          />
        ))}
      </div>
    )
  }
  
  export function ErrorBox({ message }) {
    return (
      <div className="rounded-md border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-600">
        {message}
      </div>
    )
  }