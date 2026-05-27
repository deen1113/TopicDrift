export function QueryLoading({ label = "Loading…" }: { label?: string }) {
  return (
    <div className="flex items-center justify-center p-8 text-muted-foreground">
      <span>{label}</span>
    </div>
  );
}
