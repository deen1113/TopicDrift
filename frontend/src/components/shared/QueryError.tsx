type QueryErrorProps = {
  error: unknown;
  refetch?: () => void;
};

export function QueryError({ error, refetch }: QueryErrorProps) {
  const message = error instanceof Error ? error.message : "Something went wrong.";
  return (
    <div className="flex flex-col items-center gap-3 p-8 text-destructive">
      <span>{message}</span>
      {refetch && (
        <button
          type="button"
          onClick={() => refetch()}
          className="rounded border border-border px-3 py-1 text-foreground hover:bg-muted"
        >
          Retry
        </button>
      )}
    </div>
  );
}
