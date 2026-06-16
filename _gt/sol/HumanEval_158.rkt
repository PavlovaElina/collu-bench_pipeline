#lang racket
(define (find_max words)
  (define (uniq w) (length (remove-duplicates (string->list w))))
  (foldl (lambda (w acc)
           (cond [(> (uniq w) (uniq acc)) w]
                 [(and (= (uniq w) (uniq acc)) (string<? w acc)) w]
                 [else acc]))
         (first words) (rest words)))
