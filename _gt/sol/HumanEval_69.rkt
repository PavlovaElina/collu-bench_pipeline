#lang racket
(define (search lst)
  (define h (make-hash))
  (for ([x lst]) (hash-update! h x add1 0))
  (define cands (for/list ([(k v) (in-hash h)] #:when (and (> k 0) (>= v k))) k))
  (if (null? cands) -1 (apply max cands)))
